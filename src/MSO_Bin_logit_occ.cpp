#include <RcppArmadillo.h>
#include <math.h>

#include "RNG.h"
#include "PolyaGamma.h"
#include <R_ext/Utils.h>
#include <iostream>
#include <exception>

#define pi           3.14159265358979323846  /* pi */
using namespace arma;
using namespace Rcpp;
using namespace R;
using namespace sugar;
using namespace std;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::depends("RcppArmadillo")]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
     /* Sampling from a Gaussain (mutivariate) distribution
     * output is row vector
     */
     int ncols = sigma.n_cols;
     arma::mat Y = arma::randn(n, ncols);  //1 by ncols matrix
     return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::mat mvrnormArma2(int n, arma::vec mu, arma::mat sigma) {
     /* Sampling from a Gaussain (mutivariate) distribution
     * output is column vector
     */
     int ncols = sigma.n_cols;
     arma::mat Y = arma::randn(ncols, n);  //ncols by 1 matrix

     //the output is ncols by 1
     return arma::repmat(mu, 1, n) + (arma::chol(sigma)).t()*Y;
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::mat mvrnormArma3(int n, arma::vec A, arma::mat invSigma) {
     /* Sampling from a Gaussain (mutivariate) distribution
     * output is column vector
     */
     int ncols = invSigma.n_cols;
     arma::mat Y = arma::randn(ncols, n);  //ncols by 1 matrix
     arma::vec mu = solve(invSigma, A);

     return mu + solve(arma::chol(invSigma),Y);
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::mat mvrnormArma4(int n, arma::vec A, arma::mat invSigma) {
     /* Sampling from a Gaussain (mutivariate) distribution
     * output is column vector
     */
     int ncols = invSigma.n_cols;
     //arma::mat Y = arma::randn(ncols, n);  //ncols by 1 matrix
     //arma::vec mu = solve(invSigma, A);

     //the output is ncols by 1
     //return solve(invSigma, A) + solve(arma::chol(invSigma), arma::randn(ncols, n));

     return solve(invSigma, A) + solve(arma::chol(invSigma), arma::randn(ncols, n));
}

// [[Rcpp::depends("RcppArmadillo")]]
double posterior_r3(double x, arma::vec wr, arma::vec sr){

     /*sample the components of the linear gaussian distribution
     * x = z_i_u - log(lambda_beta_i) or y_ij_u - log(lambda_alpha_i)
     * wr and sr are the constants required to approximate the
     * Logistic distribution
     */

     NumericVector w(3);

     for (int comp_j=0; comp_j<3; ++comp_j){
          w(comp_j) = wr(comp_j)*exp( -0.5*std::pow(x/sr(comp_j), 2) )/ (sr(comp_j) * sqrt(2 * pi));
     }
     w = w/sum(w);

     //writing my own sampling function here
     arma::mat u;
     int r;
     u.randu(1);

     if ( u(0)< w(0) ){
          r = 1;
     }
     else if ( u(0)< (w(0)+w(1)) )
     {
          r = 2;
     }
     else
     {
          r = 3;
     }
     return r;
}

NumericVector arma2vec(arma::vec x) {
     //converts from arma::vec to NumericVector
     return Rcpp::NumericVector(x.begin(), x.end());
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::vec invlogit(arma::mat lincomb){
     //inverse logit
     return 1.0/( 1.0 + exp( -lincomb ) );
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::vec ln_invlogit(arma::mat lincomb){
     //log inverse logit
     return -log( 1 + exp( -lincomb ) );
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::vec log_not_prob(arma::mat lincomb){
     //log(1-prob(i))
     return -log( 1 + exp( lincomb ) );
}


// [[Rcpp::depends("RcppArmadillo")]]
arma::vec rpg2(arma::mat scale) {
     /*C++-only interface to PolyaGamma class
     draws random PG variates from arma::vectors of n's and psi's
     shape = 1
     Code adapted from the BayesLogit-master github repository

     YOU NEED THE FOLLOWING FILES IN THE FOLDER: PolyaGamma.h,
     RcppExports.cpp, RNG.cpp, RNG.h, RRNG.cpp, RRNG.h

     */

     RNG r;
     PolyaGamma pg;
#ifdef USE_R
     GetRNGstate();
#endif
     int d = scale.n_elem;
     colvec result(d);
     for(int i=0; i<d; i++) {
          result[i] = pg.draw(1, scale(i), r);
     }
#ifdef USE_R
     PutRNGstate();
#endif
     return result;
}

double rpg4(double scale) {
     //draws 1 PG(1, scale) random variables
     RNG r;
     PolyaGamma pg;

     double result;
     result= pg.draw(1, scale, r);

     return result;
}

// [[Rcpp::depends("RcppArmadillo")]]
arma::vec rpg5(arma::mat scale) {
     /*C++-only interface to PolyaGamma class
     draws random PG variates from arma::vectors of n's and psi's
     shape = 1
     Code adapted from the BayesLogit-master github repository

     YOU NEED THE FOLLOWING FILES IN THE FOLDER: PolyaGamma.h,
     RcppExports.cpp, RNG.cpp, RNG.h, RRNG.cpp, RRNG.h

     */

     int d = scale.n_elem;
     colvec result(d);
     for(int i=0; i<d; i++) {
          result[i] = rpg4(scale(i));
     }

     return result;
}

// [[Rcpp::depends("RcppArmadillo")]]
double rgammadouble(int a, double b, double c)
{   //from http://iamrandom.com/rgamma-rgammadouble
     Rcpp::NumericVector x = rgamma(a,b,1.0/c);
     return x(0);
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double rnormdouble(double b, double c)
{   //from http://iamrandom.com/rnorm-rnormdouble
     //b = mean; c = sd
     Rcpp::NumericVector x = rnorm(1,b,c);
     return x(0);
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
List MSOBinlogitcpp(arma::mat X, arma::mat V, arma::mat Y,
                    double a2, double b2, double A2, double B2)
{  
  
  // define some matrices and constants
  arma::mat X_t = X.t(); //the transpose of the design matrix
  int ns = Y.n_rows; //the total number of species
  int J = Y.n_cols; //number of sites
  int nd = V.n_cols; //number of columns in detection covariates
  int no = X.n_cols; //number of columns in occupancy covariates

  int sigma2_alpha_a = (1 + ns*nd)*0.5;
  int sigma2_beta_a = (1 + ns*no)*0.5;
  double inv_A2 = 1.0/A2;
  double inv_B2 = 1.0/B2;
  
  //Some initial vaues and definitions
  double inv_nabla_alpha = 1.0;
  double inv_nabla_beta = 1.0;
  
  double tau_alpha;
  double tau_beta;
  
  double inv_scale_alpha;
  double inv_scale_beta;
  
  arma::mat alpha_quad = zeros<arma::mat>(1, 1);
  arma::mat beta_quad = zeros<arma::mat>(1, 1);
  
  arma::mat mu_alpha = zeros<arma::mat>(nd, 1);
  arma::mat alpha = zeros<arma::mat>(nd, ns);
  
  arma::mat mu_beta = zeros<arma::mat>(no, 1);
  arma::mat beta = zeros<arma::mat>(no, ns);
  
  arma::mat z;
  arma::mat omega_alpha;
  arma::mat omega_beta;
  
  //make identity matrices
  arma::mat I_nd(nd, nd);  I_nd.eye();
  arma::mat I_no(no, no);  I_no.eye();
  
  arma::mat pg_beta;
  arma::mat pg_alpha;
  
  //sampling
  
  //sample tau_alpha
  for (int i_species=0; i_species<ns; i_species++){
    alpha_quad += (alpha.col(i_species) - mu_alpha).t()*(alpha.col(i_species) - mu_alpha);
    //Rcpp::Rcout << "\n i_species = " << i_species << std::endl;
    //Rcpp::Rcout << "\n alpha_quad = " << alpha_quad << std::endl;
    //Rcpp::Rcout << "\n ------------------- " << std::endl;
  }
  
  inv_scale_alpha =  1.0/( inv_nabla_alpha + 0.5*alpha_quad(0) ) ; 
  tau_alpha = rgammadouble(1, sigma2_alpha_a, inv_scale_alpha );
  
  //sample nabla_alpha
  inv_nabla_alpha = rgammadouble(1, 0.5, 1.0/( tau_alpha + inv_A2) );
  //arma::mat RR = zeros<arma::mat>(5000, 1);
  //for (int i=0; i<5000; i++){ RR(i)= rgammadouble(1, 20, 4 );}
  
  //sample tau_beta
  for (int i_species=0; i_species<ns; i_species++){
    beta_quad += (beta.col(i_species) - mu_beta).t()*(beta.col(i_species) - mu_beta);
  }
    
  //sample nabla_beta
  inv_nabla_beta = rgammadouble(1, 0.5, 1.0/( tau_beta + inv_B2) );
  
  //sample mu_alpha
  mu_alpha = mvnrnd( (a2*tau_alpha/(1.0 + ns*a2*tau_alpha))*sum(alpha, 1), (a2/(1+ns*a2*tau_alpha))*I_nd , 1);
  //Rcpp::Rcout << "\n mu_alpha= " << mu_alpha << std::endl;
  
  //sample mu_beta
  mu_beta = mvnrnd( (b2*tau_beta/(1.0 + ns*b2*tau_beta))*sum(beta, 1), (b2/(1+ns*b2*tau_beta))*I_no , 1);
  
  //sample pg_beta
  for (int i_species=0; i_species<ns; i_species++){
    //sample from Polya-gamm variables in turn for each of the species
    pg_beta = rpg5( X*beta.col(i_species) );
    Rcpp::Rcout << "\n pg_beta= " << pg_beta << std::endl;
  }
  
  //Rcpp::Rcout << "\n row sum = " << sum(X,0) << std::endl;
  //Rcpp::Rcout << "\n col sum = " << sum(X,1) << std::endl;
  
  return List::create(_["inv_nabla_alpha"]=inv_nabla_alpha);
                      
                      
}



