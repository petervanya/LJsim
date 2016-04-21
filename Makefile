include system.mk
blddir = build
FFLAGS = -Og -fcheck=all
F2PY ?= f2py

all: lj_functions_c lj_functions_f

lj_functions_c:
	python3 setup.py build_ext --inplace

lj_functions_f:
	@mkdir -p ${blddir}
	CFLAGS="${CFLAGS}" ${F2PY} -c --build-dir ${blddir} --fcompiler=${FVENDOR} \
		   --f90exec=${FC} --f90flags="${FFLAGS}" --compiler=${CVENDOR} \
		   -m $@ ${LDFLAGS} lj_functions_f.f90

clean:
	-rm *.mod
	-rm -r build

distclean: clean
	-rm *.so
