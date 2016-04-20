module ljf

implicit none

contains

real(8) function norm(v)
    real(8), intent(in) :: v(3)

    norm = sqrt(sum(v**2))
end function

real(8) function V_LJ(mag_r, eps, sigma, rc)
    real(8), intent(in) :: mag_r, eps, sigma, rc

    real(8) :: V_rc

    V_rc = 4*eps*((sigma/rc)**12 - (sigma/rc)**6)
    if (mag_r < rc) then
        V_LJ = 4*eps*((sigma/mag_r)**12 - (sigma/mag_r)**6) - V_rc
    else
        V_LJ = 0.d0
    end if
end function

function force(r, eps, sigma, rc)
    real(8), intent(in) :: r(3), eps, sigma, rc
    real(8) :: force(3)

    real(8) :: mag_dr

    mag_dr = norm(r)
    if (mag_dr < rc) then
        force = 4*eps*(-12*(sigma/mag_dr)**12 + 6*(sigma/mag_dr)**6)*r/mag_dr**2
    else
        force = 0.d0
    end if
end function

real(8) function tot_PE(pos_list, eps, sigma, rc) result(E)
    real(8), intent(in) :: pos_list(:, :), eps, sigma, rc

    integer :: N, i, j

    E = 0.d0
    N = size(pos_list, 1)
    do i = 1, N
        do j = i+1, N
            E = E + V_LJ(norm(pos_list(i, :)-pos_list(j, :)), eps, sigma, rc)
        end do
    end do
end function

function force_list(pos_list, L, eps, sigma, rc, inv) result(F)
    real(8), intent(in) :: pos_list(:, :), L, eps, sigma, rc
    real(8) :: F(size(pos_list, 1), 3)
    external :: inv
    interface
        subroutine inv(A, n, A_inv)
            integer :: n
            real(8), intent(in) :: A(n, n)
            real(8), intent(out) :: A_inv(n, n)
        end subroutine
    end interface

    real(8) :: &
        force_mat(size(pos_list, 1), size(pos_list, 1), 3), &
        cell(3, 3), inv_cell(3, 3), dr(3), G(3), G_n(3), dr_n(3)
    integer :: N, i, j

    N = size(pos_list, 1)
    force_mat(:, :, :) = 0.d0
    cell(:, :) = 0.d0
    forall (i = 1:3) cell(i, i) = 1.d0
    cell = L*cell
    call inv(cell, 3, inv_cell)
    do i = 1, N
        do j = 1, i-1
            dr = pos_list(j, :)-pos_list(i, :)
            G = matmul(inv_cell, dr)
            G_n = G-nint(G)
            dr_n = matmul(cell, G_n)
            force_mat(i, j, :) = force(dr_n, eps, sigma, rc)
        end do
    end do
    force_mat = force_mat - reshape(force_mat, [N, N, 3], order=[2, 1, 3])
    F = sum(force_mat, 2)
end function

end module
