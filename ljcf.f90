module lj_functions_cf

use iso_c_binding

implicit none

private

integer, parameter :: dp = c_double

public :: tot_PE, force_list, c_tot_pe, c_force_list

type, public, bind(c) :: Sp_t
    real(dp) :: eps, sigma, rc, L, dt
    integer :: N, Nt, thermo, seed
    logical :: dump, use_numba, use_cython, use_fortran, use_cfortran
end type

abstract interface
    subroutine matrix_op(A_in, A_out, n)
        import :: dp
        integer, intent(in), value :: n
        real(dp), intent(in) :: A_in(n, n)
        real(dp), intent(out) :: A_out(n, n)
    end subroutine
end interface

contains

real(dp) function norm(v)
    real(dp), intent(in) :: v(3)

    norm = sqrt(sum(v**2))
end function

real(dp) function V_LJ(mag_r, sp)
    real(dp), intent(in) :: mag_r
    type(Sp_t), intent(in) :: sp

    real(dp) :: V_rc

    V_rc = 4*sp%eps*((sp%sigma/sp%rc)**12 - (sp%sigma/sp%rc)**6)
    if (mag_r < sp%rc) then
        V_LJ = 4*sp%eps*((sp%sigma/mag_r)**12 - (sp%sigma/mag_r)**6) - V_rc
    else
        V_LJ = 0.d0
    end if
end function

real(dp) function tot_PE(pos_list, sp) result(E)
    real(dp), intent(in) :: pos_list(:, :)
    type(Sp_t), intent(in) :: sp

    integer :: N, i, j

    E = 0.d0
    N = size(pos_list, 1)
    do i = 1, N
        do j = i+1, N
            E = E + V_LJ(norm(pos_list(i, :)-pos_list(j, :)), sp)
        end do
    end do
end function

real(dp) function c_tot_pe(pos_list, sp, n) result(E) bind(c, name='tot_pe')
    integer, intent(in), value :: n
    real(dp), intent(in) :: pos_list(n, 3)
    type(Sp_t), intent(in), value :: sp

    E = tot_PE(pos_list, sp)
end function

function force(r, sp)
    real(dp), intent(in) :: r(3)
    type(Sp_t), intent(in) :: sp
    real(dp) :: force(3)

    real(dp) :: mag_dr

    mag_dr = norm(r)
    if (mag_dr < sp%rc) then
        force = 4*sp%eps*(-12*(sp%sigma/mag_dr)**12 + 6*(sp%sigma/mag_dr)**6)*r/mag_dr**2
    else
        force = 0.d0
    end if
end function

function force_list(pos_list, sp, inv) result(F)
    real(dp), intent(in) :: pos_list(:, :)
    type(Sp_t), intent(in) :: sp
    real(dp) :: F(size(pos_list, 1), 3)
    interface
        function inv(A)
            import dp
            real(dp), intent(in) :: A(:, :)
            real(dp) :: inv(size(A, 2), size(A, 1))
        end function
    end interface

    real(dp) :: &
        force_mat(size(pos_list, 1), size(pos_list, 1), 3), &
        cell(3, 3), inv_cell(3, 3), dr(3), G(3), G_n(3), dr_n(3)
    integer :: N, i, j

    N = size(pos_list, 1)
    force_mat(:, :, :) = 0.d0
    cell(:, :) = 0.d0
    forall (i = 1:3) cell(i, i) = 1.d0
    cell = sp%L*cell
    inv_cell = inv(cell)
    do i = 1, N
        do j = 1, i-1
            dr = pos_list(j, :)-pos_list(i, :)
            G = matmul(inv_cell, dr)
            G_n = G-nint(G)
            dr_n = matmul(cell, G_n)
            force_mat(i, j, :) = force(dr_n, sp)
        end do
    end do
    force_mat = force_mat - reshape(force_mat, [N, N, 3], order=[2, 1, 3])
    F = sum(force_mat, 2)
end function

subroutine c_force_list(pos_list, sp, c_inv, F, n) bind(c, name='force_list')
    integer, intent(in), value :: n
    real(dp), intent(in) :: pos_list(n, 3)
    type(Sp_t), intent(in), value :: sp
    real(dp), intent(out) :: F(n, 3)
    type(c_funptr), value :: c_inv

    procedure(matrix_op), pointer :: f_inv

    call c_f_procpointer(c_inv, f_inv)
    F = force_list(pos_list, sp, inv)

    contains

    function inv(A)
        real(dp), intent(in) :: A(:, :)
        real(dp) :: inv(size(A, 2), size(A, 1))

        call f_inv(A, inv, size(A, 1))
    end function
end subroutine

end module
