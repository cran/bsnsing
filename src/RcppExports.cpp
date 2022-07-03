// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_bslearn
List rcpp_bslearn(int nrows, int ncols, IntegerVector input_y, IntegerVector input_x, IntegerVector grp, int max_rules, int max_time, int node_size, int no_same_gender_children, int verbose);
RcppExport SEXP _bsnsing_rcpp_bslearn(SEXP nrowsSEXP, SEXP ncolsSEXP, SEXP input_ySEXP, SEXP input_xSEXP, SEXP grpSEXP, SEXP max_rulesSEXP, SEXP max_timeSEXP, SEXP node_sizeSEXP, SEXP no_same_gender_childrenSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type nrows(nrowsSEXP);
    Rcpp::traits::input_parameter< int >::type ncols(ncolsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type input_y(input_ySEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type input_x(input_xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type grp(grpSEXP);
    Rcpp::traits::input_parameter< int >::type max_rules(max_rulesSEXP);
    Rcpp::traits::input_parameter< int >::type max_time(max_timeSEXP);
    Rcpp::traits::input_parameter< int >::type node_size(node_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type no_same_gender_children(no_same_gender_childrenSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_bslearn(nrows, ncols, input_y, input_x, grp, max_rules, max_time, node_size, no_same_gender_children, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bsnsing_rcpp_bslearn", (DL_FUNC) &_bsnsing_rcpp_bslearn, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_bsnsing(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}