/* The ENUM algorithm used in the bslearn function. 
This is an (optional) part of the bsnsing library for decision tree induction. 
It gives exactly the same results as the ENUM algorithm written in R, but runs faster. 
Developer: Yanchao Liu (yanchaoliu@wayne.edu)
*/
//#include <stdio.h>
//#include <stdlib.h>
#include <time.h>
//#include <R.h>
#include "Rcpp.h"
using namespace Rcpp;

#undef DEBUG 
#define MAXRULES 10
#define BLOCKSIZE 8192

typedef struct candidate_rule {
	int ncols;
  	int col_indx[MAXRULES];
  	double FP;
  	double FN;
  	double tau;
} rule_t;


IntegerVector eval_rule(int nr, IntegerVector input_y, IntegerVector input_x, IntegerVector col_idx, 
				int n_col_indx){
	int i,r;
	int correct = 0;
	int FP = 0;
	int FN = 0;
	int pred_n_neg = 0;
	int pred_n_pos = 0;
	
	for(i = 0; i < nr; i++){
		if (input_y[i] == 1){
			correct = 0;
			for(r = 0; r < n_col_indx; r++){
				if(input_x[nr*col_idx[r] + i] == 1){
					correct = 1;
					break;
				}
			}
			if(!correct){
				FN += 1;
				pred_n_neg += 1;
			} else {
				pred_n_pos += 1;
			}
		} else {
			correct = 1;
			for(r=0; r < n_col_indx; r++){
				if(input_x[nr*col_idx[r] + i] == 1){
					correct = 0;
					break;
				}
			}
			if(!correct){
				FP += 1;
				pred_n_pos += 1;
			} else {
				pred_n_neg += 1;
			}
		}
	}
	IntegerVector res = IntegerVector::create(FP,FN,pred_n_neg,pred_n_pos);
	return res;
}

//' C implementation of the bslearn function
//'
//' @param nrows number of rows
//' @param ncols number of columns
//' @param input_y Integer vector 
//' @param input_x Integer matrix
//' @param grp group membership indices
//' @param max_rules max number of rules
//' @param max_time max running time in seconds
//' @param node_size minimum node size
//' @param no_same_gender_children if TRUE child nodes must be of different predicted classes
//' @param verbose verbose level
//' @return a list containing four elements
// [[Rcpp::export]]
List rcpp_bslearn(int nrows, int ncols, 
	IntegerVector input_y, 
	IntegerVector input_x, 
	IntegerVector grp,
	int max_rules, 
	int max_time, 
	int node_size,
	int no_same_gender_children,
	int verbose
) {
	int i,j,k;
	int n0 = 0;
	int n1 = 0;
	clock_t beg_time;
	beg_time = clock();

	for(i = 0; i < nrows; i++){
		if(input_y[i] == 0){
			n0 += 1;
		} else {
			n1 += 1;
		}
	}
	
	int neval = 0;
	double vbest = n0*n1/2.0;
	int cols_best[MAXRULES];  // to store the column indexes in the best solution 
	int n_cols_best = 0;  // how many columns are there in the best solution
	int last_node_indx = 0;
	int blocksize = ncols;  // block size for the dynamic array of candidate rules
	rule_t *search_list;
	search_list = (rule_t *) malloc(blocksize*sizeof(rule_t));

	/* Create the root nodes (single-variable rules) and enter them into the list */
	for(j = 0; j < ncols; j++){
		int FP, FN, pred_n_neg, pred_n_pos;
		FP = FN = pred_n_neg = pred_n_pos = 0;
		double this_v, this_tau;
		IntegerVector this_cols (MAXRULES);
		int this_n_cols = 1;
		this_cols[0] = j;
		IntegerVector eres = eval_rule(nrows, input_y, input_x, this_cols, this_n_cols);
		FP = eres[0];
		FN = eres[1];
		pred_n_neg = eres[2];
		pred_n_pos = eres[3];
		this_v = n1*FP + n0*FN - 2.0*FP*FN;
		neval += 1;
		if(pred_n_neg < node_size || pred_n_pos < node_size){
			continue;
		}
		if(this_v < vbest){
			if(no_same_gender_children){
				/* check if the children have different majorit classes */
				int TP = n1 - FN;
				int TN = n0 - FP;
				double left1prob = 1.0*TP/(TP+FP);
				double right1prob = 1.0*FN/(FN+TN);
				if((left1prob > 0.5 && right1prob < 0.5) || (left1prob < 0.5 && right1prob > 0.5)){
					vbest = this_v;
					cols_best[0] = j;
					n_cols_best = 1;
				}
			} else {
				vbest = this_v;
				cols_best[0] = j;
				n_cols_best = 1;
			}
		}
		/* What is the best possible objval going forward from the current candidate? 
		It depends on the signs of (n0 - 2*FP) and (n1 - 2*FN). */
		bool FP_too_big = (FP >= n0/2.0);
		bool FN_too_small = (FN <= n1/2.0);
		if(FP_too_big){
			if(FN_too_small){
				this_tau = this_v;
			} else {
				this_tau = n0*(n1-FN);
			}
		} else {
			if(FN_too_small){
				this_tau = n1*FP;
			} else {
				this_tau = 0;
			}
		}
		if(this_tau < vbest){
			/* save this for further exploration */
			rule_t this_rule;
			this_rule.ncols = 1;
			this_rule.col_indx[0] = j;
			this_rule.FP = FP;
			this_rule.FN = FN;
			this_rule.tau = this_tau;
			search_list[last_node_indx] = this_rule;
			last_node_indx += 1;
		}
	}

	int memory_ok = 1;
	while(last_node_indx){
		if(search_list[0].ncols >= max_rules){
			free(search_list);
			last_node_indx = 0;
			break;
		}
		rule_t *next_level_list = (rule_t *) malloc(BLOCKSIZE*sizeof(rule_t));
		int n_elem_next_level_list = 0;
		/* iterate through this level search list */
		for (k = 0; k < last_node_indx; k++){
			rule_t cur_node = search_list[k];
			if(cur_node.tau >= vbest){
				continue;
			}
			/* evaluate each 1-augment candidate of the cur_node */
			for(j = (cur_node.col_indx)[cur_node.ncols-1]+1; j < ncols; j++){
				/* candidate column must not be from the same group as any of the existing columns, 
					and index must go up, to avoid redundant evaluations */
				int j1;
				int pass_this_j = 0;
				for(j1 = 0; j1 < cur_node.ncols; j1++){
					if(grp[(cur_node.col_indx)[j1]] == grp[j]) {
						pass_this_j = 1;
						break;
					}
				}
				if(pass_this_j) continue;
				IntegerVector this_cols (MAXRULES);
				int this_n_cols;
				int FP, FN, pred_n_neg, pred_n_pos;
				FP = FN = pred_n_neg = pred_n_pos = 0;
				double this_v, this_tau;
				for(j1 = 0; j1 < cur_node.ncols; j1++){
					this_cols[j1] = (cur_node.col_indx)[j1];
				}
				this_cols[cur_node.ncols] = j;
				this_n_cols = cur_node.ncols + 1;
				IntegerVector eres = eval_rule(nrows, input_y, input_x, this_cols, this_n_cols);
				FP = eres[0];
				FN = eres[1];
				pred_n_neg = eres[2];
				pred_n_pos = eres[3];
				neval += 1;
				if(pred_n_neg < node_size || pred_n_pos < node_size){
					continue;
				}
				if(FP == cur_node.FP && FN == cur_node.FN){
					continue;
				}
				this_v = n1*FP + n0*FN - 2.0*FP*FN;
				if(this_v < vbest){
					if(no_same_gender_children){
						/* check if the children have different majorit classes */
						int TP = n1 - FN;
						int TN = n0 - FP;
						double left1prob = 1.0*TP/(TP+FP);
						double right1prob = 1.0*FN/(FN+TN);
						if((left1prob > 0.5 && right1prob < 0.5) || (left1prob < 0.5 && right1prob > 0.5)){
							vbest = this_v;
							for(j1=0; j1 < this_n_cols; j1++){
								cols_best[j1] = this_cols[j1];
							}
							n_cols_best = this_n_cols;
						}
					} else {
						vbest = this_v;
						for(j1=0; j1 < this_n_cols; j1++){
							cols_best[j1] = this_cols[j1];
						}
						n_cols_best = this_n_cols;
					}
				}
				bool FP_too_big = (FP >= n0/2.0);
				bool FN_too_small = (FN <= n1/2.0);
				if(FP_too_big){
					if(FN_too_small){
						this_tau = this_v;
					} else {
						this_tau = n0*(n1-FN);
					}
				} else {
					if(FN_too_small){
						this_tau = n1*FP;
					} else {
						this_tau = 0;
					}
				}
				if(this_tau < vbest){
					/* save this for further exploration */
					rule_t this_rule;
					this_rule.ncols = this_n_cols;
					for(j1=0; j1<this_n_cols; j1++){
						this_rule.col_indx[j1] = this_cols[j1];
					}
					this_rule.FP = FP;
					this_rule.FN = FN;
					this_rule.tau = this_tau;
					/* before add to the array, check if the array is big enough; if not, expand */
					n_elem_next_level_list += 1;
					if(n_elem_next_level_list % BLOCKSIZE == 0){
						int n_blocks = n_elem_next_level_list / BLOCKSIZE;
						rule_t *bigger_next_level_list = (rule_t *) realloc(next_level_list, (n_blocks+1)*BLOCKSIZE*sizeof(rule_t));
						if(bigger_next_level_list == NULL){
							free(next_level_list);
							n_elem_next_level_list = 0;
							memory_ok = 0;
							break;
						} else {
							next_level_list = bigger_next_level_list;
						}
					}
					next_level_list[n_elem_next_level_list-1] = this_rule;
				}
			}
			if(!memory_ok) break;
		}
		free(search_list);
		search_list = next_level_list;
		last_node_indx = n_elem_next_level_list;
		/* if time limit is reached, terminate */
		if((double)(clock() - beg_time)/CLOCKS_PER_SEC > max_time){
			free(search_list);
			last_node_indx = 0;
		}
	}
	int sol_n_cols = n_cols_best;
	IntegerVector sol_cols (n_cols_best);
	for(j = 0; j < n_cols_best; j++){
		sol_cols[j] = cols_best[j];
	}
	double sol_vbest = vbest;
	List bslearn_result = List::create(Named("sol_cols", sol_cols), Named("sol_n_cols", sol_n_cols), Named("sol_vbest", sol_vbest), Named("neval", neval));
	return bslearn_result;
}

