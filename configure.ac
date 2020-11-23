m4_include([aux/ax_check_zlib.m4])
m4_include([aux/ax_openmp.m4])
m4_include([aux/ax_cxx_compile_stdcxx.m4])
# m4_include([aux/ax_boost_base.m4])

AC_INIT([mmutil],[0.2.1],[ypp@stat.ubc.ca])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE

AC_CHECK_HEADER_STDBOOL
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T

AX_CXX_COMPILE_STDCXX([14], noext, mandatory)

# AX_CXX_COMPILE_STDCXX([17], noext, mandatory)
# user_LDFLAGS="${user_LDFLAGS} -ltbb" # to use intel TBB

# Checks for programs.
AC_PROG_CXX
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AC_PROG_INSTALL

# Checks for header files.
AC_LANG_PUSH([C++])
AC_CHECK_HEADERS([eigen3/Eigen/Dense])
AC_CHECK_HEADERS([eigen3/Eigen/Sparse])
AC_OPENMP
AC_LANG_POP([C++])

AC_CHECK_HEADERS([zlib.h])

# Checks for library functions.
AC_CHECK_LIB([m],[exp])
AX_CHECK_ZLIB([],[])
# AX_BOOST_BASE([1.65.0])

AM_INIT_AUTOMAKE([-Wall foreign subdir-objects])

CXXFLAGS="$CXXFLAGS"
CFLAGS="$CFLAGS"

AC_ARG_ENABLE([debug],
              [AS_HELP_STRING([--enable-debug],
	                      [enable debugging info (default is no)])],
              [], [enable_debug=no])

AC_ARG_ENABLE([optim],
              [AS_HELP_STRING([--enable-optim@<:@=0|1|2|3@:>@],
	                      [set optimization level (default is 3)])],
              [if test "x$enable_optim" = xyes; then enable_optim=3; fi],
              [enable_optim=3])

AC_ARG_ENABLE([vectorize],
              [AS_HELP_STRING([--enable-vectorize],
	                      [enable vectorization (default is yes)])],
              [], [enable_vectorize=yes])

AS_IF([test "x$enable_optim" != xno], 
	    [ext_CFLAGS="$ext_CFLAGS -O$enable_optim"])

AS_IF([test "x$enable_vectorize" = xyes],
      [vec_CFLAGS="-msse2"],
      [vec_CFLAGS="-DEIGEN_DONT_VECTORIZE"])

AS_IF([test "x$enable_debug" = xyes],
      [debug_CFLAGS="-g -Wall -DDEBUG"],
      [debug_CFLAGS="-DNDEBUG"])

CFLAGS="${generic_CFLAGS} ${ext_CFLAGS} ${user_CFLAGS} ${debug_CFLAGS} ${vec_CFLAGS}"
CXXFLAGS="${CFLAGS} ${OPENMP_CXXFLAGS}"
LDFLAGS="${LDFLAGS} ${user_LDFLAGS}"

# makefiles to configure
AC_CONFIG_FILES([Makefile src/Makefile])

# make it happen
AC_OUTPUT
