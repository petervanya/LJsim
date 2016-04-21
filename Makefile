include system.mk
blddir = build
FFLAGS ?= -O3
F2PY ?= f2py

all: lj_functions_c lj_functions_f ljcf.so

ljcf.so: ljcf.f90
	${FC} -shared -fPIC ${FFLAGS} -o $@ $^

lj_functions_c:
	python3 setup.py build_ext --inplace

lj_functions_f:
	@mkdir -p ${blddir}
	CFLAGS="${CFLAGS}" ${F2PY} -c --build-dir ${blddir} --fcompiler=${FVENDOR} \
		   --f90exec=${FC} --f90flags="${FFLAGS}" --compiler=${CVENDOR} \
		   -m $@ ${LDFLAGS} $@.f90

clean:
	-rm *.mod
	-rm -r build
	-rm *.so
	-rm -r *.so.dSYM
	-rm *.c
	-rm -r __pycache__

distclean: clean
	-rm *.so
