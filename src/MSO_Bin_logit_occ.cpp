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

#ifdef _OPENMP
  #include <omp.h>
#endif

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
double invlogit_d(arma::mat lincomb){
  //inverse logit
  return 1.0/( 1.0 + exp( -lincomb(0) ) );
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

double rpgK4(int K, double scale) {
  //draws 1 PG(K, scale) random variables
  RNG r;
  PolyaGamma pg;
  
  double result;
  result= pg.draw(K, scale, r);
  
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
arma::vec rpgK5(arma::mat K, arma::mat scale) {
  /*C++-only interface to PolyaGamma class
  draws random PG variates from arma::vectors of n's and psi's
  shape = K
  Code adapted from the BayesLogit-master github repository
  
  YOU NEED THE FOLLOWING FILES IN THE FOLDER: PolyaGamma.h,
  RcppExports.cpp, RNG.cpp, RNG.h, RRNG.cpp, RRNG.h
  
  */
  
  int d = scale.n_elem;
  colvec result(d);
  for(int i=0; i<d; i++) {
    result[i] = rpgK4(K(i),scale(i));
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


// [[Rcpp::export]]  
double chooseC(int n, int k) {
  return Rf_choose(n, k);
}

// [[Rcpp::export]]  
double binom_mass(int y, double psi, double p, int K){
  //calculate the mass function for a binomial distribution using MSO model
  double temp=0.0;
  
  if (y==0){
    temp = 1-psi;
  }
  
  //temp += Rf_choose(K, y)*pow(p,y)*pow(1-p, K-y)*psi;
  temp += Rf_choose(K, y)*exp(y*log(p) + (K-y)*log(1-p))*psi;
  
  return(temp);
}

// [[Rcpp::export]]  
double lbinom_mass(int y, double psi, double p, int K){
  //calculate the log mass function for a binomial distribution using MSO model
  return(log(binom_mass(y, psi, p, K)));
}

// [[Rcpp::export]]  
double lbinom_mass_p2(int y, double psi, double p, int K){
  //calculate the log mass function raised to power 2
  //for a binomial distribution using MSO model
  return(pow(lbinom_mass(y, psi, p, K), 2));
}  
  
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
List MSOBinlogitcpp(arma::mat X, arma::mat V, arma::mat Y, arma::mat z,
                    arma::mat p, arma::mat psi,
                    arma::mat nsitevisits,
                    double a2, double b2, double A2, double B2,
                    int ndraws, double percent_burn_in, int thin,
                    int selection)
{  
  /* Undertakes sampling from a MSO model with known species richness!
   * Some model selection statistics are computed as well if required.
   */
  
  // define some matrices and constants
  arma::mat X_t = X.t(); //the transpose of the design matrix
  int ns = Y.n_rows; //the total number of species
  int J = Y.n_cols; //number of sites
  int nd = V.n_cols; //number of columns in detection covariates
  int no = X.n_cols; //number of columns in occupancy covariates
  
  NumericVector siteindex(J); //siteindex = 1, 2, ..., J
  for (int idown =0; idown<J; idown++){siteindex(idown) = idown+1;}
  
  //-----------------------------------------------------------------------
  //sample sigma2 parameters and nabla parameters
  double sigma2_alpha_a = (1.0 + ns*nd)*0.5;
  double sigma2_beta_a = (1.0 + ns*no)*0.5;
  double inv_A2 = 1.0/A2;
  double inv_B2 = 1.0/B2;

/*
Rcpp::Rcout << "\n sigma2_alpha_a = " << sigma2_alpha_a << std::endl;
Rcpp::Rcout << "\n sigma2_beta_a = " << sigma2_beta_a << std::endl;
Rcpp::Rcout << "\n inv_A2 = " << inv_A2 << std::endl;
Rcpp::Rcout << "\n inv_B2 = " << inv_B2 << std::endl;
*/

  //Some initial values and definitions
  double inv_nabla_alpha = rgammadouble(1, 0.5, inv_A2);
  double inv_nabla_beta = rgammadouble(1, 0.5, inv_B2);
  
  double tau_alpha; //1/sigma2(alpha)
  double tau_beta;  //1/sigma2(beta)
  
  //double inv_scale_alpha;
  //double inv_scale_beta;
  double scale_alpha;
  double scale_beta;
  
  arma::mat alpha_quad = zeros<arma::mat>(1, 1);
  arma::mat beta_quad = zeros<arma::mat>(1, 1);
  //-----------------------------------------------------------------------

  //sample mu parameters
  arma::mat mu_alpha = zeros<arma::mat>(nd, 1);
  arma::mat mu_beta = zeros<arma::mat>(no, 1);
  
  //make identity matrices
  arma::mat I_nd(nd, nd);  I_nd.eye();
  arma::mat I_no(no, no);  I_no.eye();
  
  //some initial values
  mu_alpha = mvnrnd( zeros<arma::mat>(nd, 1) , I_nd);
  mu_beta = mvnrnd( zeros<arma::mat>(no, 1) , I_no);
  //-----------------------------------------------------------------------
  
  //sample alpha and beta matrices
  arma::mat alpha = zeros<arma::mat>(nd, ns);
  arma::mat beta = zeros<arma::mat>(no, ns);
  
  //some initial values
  for (int i_species=0; i_species<ns; i_species++){ alpha.col(i_species) = mvnrnd( zeros<arma::mat>(nd, 1) , I_nd);}
  for (int i_species=0; i_species<ns; i_species++){ beta.col(i_species) = mvnrnd( zeros<arma::mat>(no, 1) , I_no);}
  
  arma::mat mu_alpha_i;
  arma::mat mu_beta_i;
  arma::mat cov_alpha_i;
  arma::mat cov_beta_i;
  
  arma::mat beta_i;
  arma::mat alpha_i;
  
  arma::mat V_iter;
  arma::mat Y_iter;
  
  arma::mat Y_temp;
  arma::mat Ytemp(ns, 1);
  
  //-----------------------------------------------------------------------

  //sample omega latent variables
  arma::mat pg_beta;
  arma::mat pg_alpha;

  //-----------------------------------------------------------------------
  
  //for sampling from z
  //dimensions of z is number of sites by number of species
  uvec z_equals1_rows;//identify all indices with z==1
  arma::vec prob(1);
  NumericVector zdraw(1);
  //-----------------------------------------------------------------------

/*    
  Rcpp::Rcout << "\n no = " << no << std::endl;
  Rcpp::Rcout << "\n nd = " << nd << std::endl;
  
  Rcpp::Rcout << "\n nrow(beta) = " << beta.n_rows << std::endl;
  Rcpp::Rcout << "\n ncol(beta) = " << beta.n_cols << std::endl;
  Rcpp::Rcout << "\n ns = " << ns << std::endl;
  
  Rcpp::Rcout << "\n nrow(alpha) = " << alpha.n_rows << std::endl;
  Rcpp::Rcout << "\n ncol(alpha) = " << alpha.n_cols << std::endl;
  
  Rcpp::Rcout << "\n dim(z) = " << z.size() << std::endl;
  Rcpp::Rcout << "\n nrow(z) = " << z.n_rows << std::endl;
  Rcpp::Rcout << "\n ncol(z) = " << z.n_cols << std::endl;
  
  Rcpp::Rcout << "\n nrow(V) = " << V.n_rows << std::endl;
  Rcpp::Rcout << "\n ncol(V) = " << V.n_cols << std::endl;
  
  Rcpp::Rcout << "\n nrow(X) = " << X.n_rows << std::endl;
  Rcpp::Rcout << "\n ncol(X) = " << X.n_cols << std::endl;
  
  Rcpp::Rcout << "\n nrow(Y) = " << Y.n_rows << std::endl;
  Rcpp::Rcout << "\n ncol(Y) = " << Y.n_cols << std::endl;
 */
  //-----------------------------------------------------------------------
    
  //The outputs
  int isamples_counter;
  int num_burnin = floor(ndraws*percent_burn_in);
  //int num_samples_kept = ndraws - num_burnin;
  int num_samples_kept = floor( (ndraws - num_burnin)/thin );
//Rcout << "\n num_burnin"   << num_burnin << std::endl;
//Rcout << "\n num_samples_kept"   << num_samples_kept << std::endl;
//Rcout << "\n last"   << num_burnin + thin*num_samples_kept << std::endl;
  
  arma::mat post_tau_alpha(1 , num_samples_kept);
  arma::mat post_tau_beta(1, num_samples_kept);
  
  arma::mat post_mu_alpha(mu_alpha.n_rows, num_samples_kept);
  arma::mat post_mu_beta(mu_beta.n_rows, num_samples_kept);
  
  //the alpha matrix that is being kept
  cube alpha_array(nd, ns , num_samples_kept);
  
  //the beta matrix that is being kept
  cube beta_array(no, ns , num_samples_kept);
  
  //the z matrix that is being kept
  cube z_array(z.n_rows, z.n_cols , num_samples_kept);
  
  //the detection probs matrix that is being kept
  cube p_array(J, ns , num_samples_kept);
  //arma::mat p_store = zeros<arma::mat>(J, ns);
  //Rcout << "p =" << p_store << std::endl;
  
  //the occupancy prob matrix that is being kept
  cube psi_array(J, ns , num_samples_kept);
  //arma::mat psi_store = zeros<arma::mat>(J, ns);
  //Rcout << "psi_store = " << psi_store << std::endl;
  arma::mat psi_conditional = psi;
  //-----------------------------------------------------------------------
  
  // model selection variables
  double elppd = 0.0;
  double pdWAIC = 0.0;
  double WAIC = 0.0;
  double CPO = 0.0;
  double binom_mass_temp;
  
  arma::mat deviance_s(1 , num_samples_kept);
  arma::mat deviance_s_tilde(1 , num_samples_kept);
  arma::mat sum_pmf_binom = zeros<arma::mat>(ns, J);
  arma::mat var_pmf_binom1 = sum_pmf_binom; //zeros<arma::mat>(ns, J);
  arma::mat var_pmf_binom2 = sum_pmf_binom; //zeros<arma::mat>(ns, J);
  arma::mat ytilde = Y; //simulated Y for goodness of fit
  arma::mat ztilde = z; //simulated z for goodness of fit
  arma::mat Ds_mat = sum_pmf_binom; //zeros<arma::mat>(ns, J); 
  arma::mat Ds_tilde_mat = sum_pmf_binom; //zeros<arma::mat>(ns, J);
  arma::mat CPO_ij =  sum_pmf_binom; //zeros<arma::mat>(ns, J);
  NumericVector ydraw(1);

  //-----------------------------------------------------------------------
  
  int thin_index = 0;
  int isamples_counter_i = 0;
  
  //sampling
  //now do the sampling here
  //for (int isamples=0; isamples<ndraws; isamples++){  
  for (int isamples=0; isamples<(num_burnin + thin*num_samples_kept); isamples++){
    
    //add in an interuptor. i.e. escape if the user cancels operations
    //checks every 1000 iterations
    if (isamples % 1000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    //sample counter
    isamples_counter = isamples - num_burnin;
    
    //sample tau_alpha
    alpha_quad(0) = 0;
    
    for (int i_species = 0; i_species < ns; i_species++){
      alpha_quad += (alpha.col(i_species) - mu_alpha).t()*(alpha.col(i_species) - mu_alpha);
    }
    //loop might be quicker!
    //alpha_quad = accu(square(alpha)) - 2*sum(alpha,1).t()*mu_alpha + ns*mu_alpha.t()*mu_alpha ;
     
    //inv_scale_alpha =  1.0/( inv_nabla_alpha + 0.5*alpha_quad(0) ) ; 
    //tau_alpha = rgammadouble(1, sigma2_alpha_a, inv_scale_alpha );
    
    //scale_alpha =  inv_nabla_alpha + 0.5*alpha_quad(0)  ; 
    //tau_alpha = rgammadouble(1, sigma2_alpha_a, scale_alpha );
    
    tau_alpha = rgammadouble(1, sigma2_alpha_a, inv_nabla_alpha + 0.5*alpha_quad(0) );
    //----------------------------------------------------------------------------
    
    //sample nabla_alpha
    inv_nabla_alpha = rgammadouble(1, 0.5, tau_alpha + inv_A2 );
    //----------------------------------------------------------------------------
    
    //sample tau_beta
    beta_quad(0) = 0;
    for (int i_species = 0; i_species < ns; i_species++){
      beta_quad += (beta.col(i_species) - mu_beta).t()*(beta.col(i_species) - mu_beta);
    }
    //beta_quad = accu(square(beta)) - 2*sum(beta,1).t()*mu_beta + ns*mu_beta.t()*mu_beta ;
    
    //inv_scale_beta =  1.0/( inv_nabla_beta + 0.5*beta_quad(0) ) ; 
    //tau_beta = rgammadouble(1, sigma2_beta_a, inv_scale_beta );
    
    //scale_beta =  inv_nabla_beta + 0.5*beta_quad(0)  ; 
    //tau_beta = rgammadouble(1, sigma2_beta_a, scale_beta );
    
    tau_beta = rgammadouble(1, sigma2_beta_a, inv_nabla_beta + 0.5*beta_quad(0) );
    //----------------------------------------------------------------------------
      
    //sample nabla_beta
    inv_nabla_beta = rgammadouble(1, 0.5, tau_beta + inv_B2 );
    //----------------------------------------------------------------------------
    
    //sample mu_alpha
    //sum(alpha, 1): alpha is a matrix of dimension nd by ns. sum across the rows.
    //I_nd is diagonal matrix of dimension n_d by n_d
    //mu_alpha elements could be sampled separately due to independence 
      mu_alpha = mvnrnd( (a2*tau_alpha/(1.0 + ns*a2*tau_alpha))*sum(alpha, 1), (a2/(1.0 + ns*a2*tau_alpha))*I_nd , 1);
    //----------------------------------------------------------------------------
    
    //sample mu_beta
    mu_beta = mvnrnd( (b2*tau_beta/(1.0 + ns*b2*tau_beta))*sum(beta, 1), (b2/(1.0 + ns*b2*tau_beta))*I_no , 1);
    //----------------------------------------------------------------------------
    
    //sample beta_i
    //this has to be stored in a big matrix!
    for (int i_species = 0; i_species < ns; i_species++){
      
      //sample from Polya-gamm variables in turn for each of the species
      pg_beta = rpg5( X*beta.col(i_species) );
      
      cov_beta_i = inv_sympd( tau_beta*I_no + X_t*diagmat( pg_beta )*X ); //arma::mat
      mu_beta_i =  cov_beta_i*(X_t*( z.col(i_species) - 0.5 ) + tau_beta*mu_beta); // arma::mat

      //sample of the occupancy regression effect for species i
      //beta_i = mvrnormArma2(1, mu_beta_i, cov_beta_i); //arma::mat
      //enter beta_i in beta column by column
      //beta.col(i_species) = beta_i;
      
      beta.col(i_species) = mvrnormArma2(1, mu_beta_i, cov_beta_i);
    }
    
    for ( int i_species = 0; i_species < ns; i_species++){
      
      //find the elements of z associated with z==1
      z_equals1_rows = find(z.col(i_species)==1);  //row number as specified by c++. if an element say 0 ==> row 0 of z is 1.
      V_iter = V.rows(z_equals1_rows);
      pg_alpha  = rpgK5( nsitevisits(z_equals1_rows),  V_iter*alpha.col(i_species) );
      
      cov_alpha_i = inv_sympd( tau_alpha*I_nd + V_iter.t()*diagmat(pg_alpha)*V_iter );
      
      //inefficient
      Y_temp = Y.row(i_species); //viewed as a row vector
  
      Y_iter = Y_temp.cols(z_equals1_rows);
      mu_alpha_i =  cov_alpha_i*(V_iter.t()*( Y_iter.t() - 0.5*nsitevisits(z_equals1_rows) ) + tau_alpha*mu_alpha); // arma::mat
      
      alpha.col(i_species) = mvrnormArma2(1, mu_alpha_i, cov_alpha_i);
    }
    
    //calculate p and psi
    //unconditional p and psi
    //use the invlogit function later!
    for (int i_species=0; i_species<ns; i_species++){
      p.col(i_species) = 1.0/(1.0 + exp( -V*alpha.col(i_species)) ); 
      psi.col(i_species) = 1.0/(1.0 + exp( -X*beta.col(i_species)) );
    }
    //Rcout << "p =" << p << std::endl;
    
    //sample from z
    for (int i_species=0; i_species<ns; i_species++){
      for (int i_sites=0; i_sites<J; i_sites++){
        
        //note Y and z are defined differently
        if (Y(i_species, i_sites) == 0){
          
          prob = 1.0/( 1.0 + (1.0 - psi(i_sites, i_species) )/( psi(i_sites, i_species)*pow(1-p(i_sites, i_species), nsitevisits(i_sites) )) );
          psi_conditional(i_sites, i_species) = prob(0); //store the conditional prob for goodness of fit work.
          zdraw = rbinom(1,1, prob(0));
          z(i_sites, i_species) = zdraw(0);
          
          //for goodness of fit
          if (zdraw(0)==0){
            ytilde(i_species, i_sites)=0;
          }else{
            ydraw = rbinom(1, nsitevisits(i_sites), p(i_sites, i_species));
            ytilde(i_species, i_sites) = ydraw(0);
          }//end goodness of fit
          
          
        }else{
          ydraw = rbinom(1, nsitevisits(i_sites), p(i_sites, i_species));
          ytilde(i_species, i_sites) = ydraw(0);
          //Rcout << "ytilde=" << ytilde(i_species, i_sites) << std::endl;
        }//endif
      }//endif
    }//end sampling of z
    
    //store the samples
    
    if ( (isamples_counter>=0) && (isamples_counter==thin_index) ){
      post_mu_alpha.col(isamples_counter_i) = mu_alpha;
      post_mu_beta.col(isamples_counter_i) = mu_beta;
      
      post_tau_alpha.col(isamples_counter_i) = tau_alpha;
      post_tau_beta.col(isamples_counter_i) = tau_beta;
      
      alpha_array.slice(isamples_counter_i) = alpha;
      beta_array.slice(isamples_counter_i) = beta;
      z_array.slice(isamples_counter_i) = z;
      
      //store p and the conditional psi matrices for each iteration kept
      p_array.slice(isamples_counter_i) = p;
      psi_array.slice(isamples_counter_i) = psi_conditional;
      
      //-------------------start of model selection calculations----------------
      
      for (int i_species=0; i_species<ns; i_species++){
        for (int i_sites=0; i_sites<J; i_sites++){
          
          //think of how to make this more efficient
          //simulated deviance
          Ds_tilde_mat(i_species, i_sites) = lbinom_mass(ytilde(i_species, i_sites), 
                       psi_conditional(i_sites,i_species),  p(i_sites,i_species), 
                       nsitevisits(i_sites));
          
          //the observed deviation
          Ds_mat(i_species, i_sites) = lbinom_mass(Y(i_species, i_sites), 
                 psi_conditional(i_sites,i_species),  p(i_sites,i_species), 
                 nsitevisits(i_sites));
          
          var_pmf_binom2(i_species, i_sites) += Ds_mat(i_species, i_sites);
          
          /*var_pmf_binom2(i_species, i_sites) += lbinom_mass(Y(i_species, i_sites), 
           psi_conditional(i_sites,i_species),  p(i_sites,i_species), 
          nsitevisits(i_sites));*/
          
          binom_mass_temp = binom_mass(Y(i_species, i_sites), 
                     psi_conditional(i_sites,i_species),  p(i_sites,i_species), 
                     nsitevisits(i_sites));
          
          sum_pmf_binom(i_species, i_sites) += binom_mass_temp;
          
          CPO_ij(i_species, i_sites) += pow(binom_mass_temp, -1);
          
          var_pmf_binom1(i_species, i_sites) += lbinom_mass_p2(Y(i_species, i_sites), 
                         psi_conditional(i_sites,i_species),  p(i_sites,i_species), 
                         nsitevisits(i_sites));
          
        }//end i_sites
      }//end i_species
      
      //calculation of the deviance and simulated deviance
      deviance_s(isamples_counter_i) = -2*accu(Ds_mat);
      deviance_s_tilde(isamples_counter_i) = -2*accu(Ds_tilde_mat);
  
      //calculation of the CPO
      CPO = -log(isamples_counter_i+1)*J*ns - accu( log(CPO_ij) );
      
      thin_index += thin;
      isamples_counter_i += 1; 
    }//end of if statement
  }//end sampling
  
  //deviance_s = -2*accu(var_pmf_binom2);
  elppd = accu( log(sum_pmf_binom) ) -log(isamples_counter_i+1)*J*ns; //correct
  var_pmf_binom2 = pow(var_pmf_binom2,2)/(isamples_counter_i+1);  //correct
  pdWAIC = accu( (var_pmf_binom1 - var_pmf_binom2)/isamples_counter_i );  //correct
  WAIC = -2*(elppd - pdWAIC);
  
  //Calculate the Bayesian p-value
  int tmp=0;
  for (int ideviance=0; ideviance<deviance_s.n_cols; ideviance++){
    if (deviance_s(ideviance)>deviance_s_tilde(ideviance)){ 
      tmp += 1;
    }else{
      tmp += 0;
    }
  }//endif
  //-------------------end of model selection calculations----------------------
  
  if (selection==1){
    //only return the model selection stats
    
    return List::create(_["BPValue"]=tmp*1.0/deviance_s.n_cols,
                        _["WAIC"]=WAIC,
                        _["CPO"]=CPO);
  }else {
    //return everything
    return List::create(_["mu_alpha"]=post_mu_alpha,
                        _["mu_beta"]=post_mu_beta,
                        _["tau_alpha"]=post_tau_alpha,
                        _["tau_beta"]=post_tau_beta,
                        _["alpha"]=alpha_array,
                        _["beta"]=beta_array,
                        _["z"]=z_array,
                        _["psi_array"]=psi_array,
                        _["p_array"]=p_array,
                        _["deviance"]=deviance_s,
                        _["deviance_tilde"]=deviance_s_tilde,
                        _["BPValue"]=tmp*1.0/deviance_s.n_cols,
                        _["WAIC"]=WAIC,
                        _["CPO"]=CPO);
  }
}

// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
List MSOBinocclogitcpp(arma::mat X, arma::mat V, arma::mat Y, arma::mat z,
                       arma::mat K, arma::mat Minv,
                       arma::mat p, arma::mat psi,
                       arma::mat tau_i, double a_tau, double b_tau,
                       arma::mat nsitevisits,
                       double a2, double b2, double A2, double B2,
                       int ndraws, double percent_burn_in, int thin)
{  
  //Undertakes sampling from a MSO model with known species richness!
  //This model attempts to account for spatial autocorrelation in the 
  //occupancy process
  
  /* Some Rcpp notes
   * Some R translations put in brackets at times.
   *
   * 1. Indices
   * arma::mat x;
   * x.row(0) is the first row a matrix. (x[1,])
   * x.row(1) is the second row a matrix. (x[2,])
   * Indices start at 0!
   *
   * 2. Multiple rows
   * x.rows(1,3) is x[2:4, ] using the R notation.
   */
  
  // define some matrices and constants
  arma::mat X_t = X.t(); //the transpose of the design matrix
  int ns = Y.n_rows; //the total number of species
  int J = Y.n_cols; //number of sites
  int nd = V.n_cols; //number of columns in detection covariates
  int no = X.n_cols; //number of columns in occupancy covariates
  int r = K.n_cols; //the number of column of K; the number of spatial random effects added
  arma::mat K_t = K.t(); //the transpose of the spatial design matrix
  
  NumericVector siteindex(J); //siteindex = 1, 2, ..., J
  for (int idown =0; idown<J; idown++){siteindex(idown) = idown+1;}
  
  //-----------------------------------------------------------------------
  //sample sigma2 parameters and nabla parameters
  double sigma2_alpha_a = (1.0 + ns*nd)*0.5;
  double sigma2_beta_a = (1.0 + ns*no)*0.5;
  double inv_A2 = 1.0/A2;
  double inv_B2 = 1.0/B2;
  
  //Some initial values and definitions
  double inv_nabla_alpha = rgammadouble(1, 0.5, inv_A2);
  double inv_nabla_beta = rgammadouble(1, 0.5, inv_B2);
  
  double tau_alpha; //1/sigma2(alpha)
  double tau_beta;  //1/sigma2(beta)
  
  //double inv_scale_alpha;
  //double inv_scale_beta;
  double scale_alpha;
  double scale_beta;
  
  arma::mat alpha_quad = zeros<arma::mat>(1, 1);
  arma::mat beta_quad = zeros<arma::mat>(1, 1);
  
  //-----------------------------------------------------------------------
  
  //sample mu parameters
  arma::mat mu_alpha = zeros<arma::mat>(nd, 1);
  arma::mat mu_beta = zeros<arma::mat>(no, 1);
  
  //make identity matrices
  arma::mat I_nd(nd, nd);  I_nd.eye();
  arma::mat I_no(no, no);  I_no.eye();
  
  //some initial values
  mu_alpha = mvnrnd( zeros<arma::mat>(nd, 1) , I_nd);
  mu_beta = mvnrnd( zeros<arma::mat>(no, 1) , I_no);

  //-----------------------------------------------------------------------
  
  //sample alpha and beta matrices
  arma::mat alpha = zeros<arma::mat>(nd, ns);
  arma::mat beta = zeros<arma::mat>(no, ns);
  arma::mat Xs_beta; //X*beta
  
  //some initial values
  for (int i_species=0; i_species<ns; i_species++){ alpha.col(i_species) = mvnrnd( zeros<arma::mat>(nd, 1) , I_nd);}
  for (int i_species=0; i_species<ns; i_species++){ beta.col(i_species) = mvnrnd( zeros<arma::mat>(no, 1) , I_no);}
  
  arma::mat mu_alpha_i;
  arma::mat mu_beta_i;
  arma::mat cov_alpha_i;
  arma::mat cov_beta_i;
  
  arma::mat beta_i;
  arma::mat alpha_i;
  
  arma::mat V_iter;
  arma::mat Y_iter;
  
  arma::mat Y_temp;
  arma::mat Ytemp(ns, 1);
  
  //-----------------------------------------------------------------------
  
  //sample omega latent variables
  arma::mat pg_beta;
  arma::mat pg_alpha;
  //-----------------------------------------------------------------------
  
  //Posterior sampling of theta
  arma::mat Ks_theta; //Ks*theta
  arma::mat mu_theta_i;
  arma::mat cov_theta_i;
  arma::mat theta = zeros<arma::mat>(r, ns);
  //-----------------------------------------------------------------------
  
  //Posterior sampling of tau_i
  double tau=1.0; //initialize tau =1; the spatial precision scalar
  mat inv_scale;
  //-----------------------------------------------------------------------
  
  //for sampling from z
  //dimensions of z is number of sites by number of species
  uvec z_equals1_rows;//identify all indices with z==1
  arma::vec prob(1);
  NumericVector zdraw(1);
  //-----------------------------------------------------------------------
  
  //The outputs
  int isamples_counter;
  int num_burnin = floor(ndraws*percent_burn_in);
  int num_samples_kept = floor( (ndraws - num_burnin)/thin );
//Rcout << "\n num_samples_kept"   << num_samples_kept << std::endl;

  arma::mat post_tau_alpha(1 , num_samples_kept);
  arma::mat post_tau_beta(1, num_samples_kept);
  arma::mat post_tau_i(ns, num_samples_kept);
  
  arma::mat post_mu_alpha(mu_alpha.n_rows, num_samples_kept);
  arma::mat post_mu_beta(mu_beta.n_rows, num_samples_kept);
  
  //the alpha matrix that is being kept
  cube alpha_array(nd, ns , num_samples_kept);
  
  //the beta matrix that is being kept
  cube beta_array(no, ns , num_samples_kept);
  
  //the z matrix that is being kept
  cube z_array(z.n_rows, z.n_cols , num_samples_kept);
  
  //the p and psi matrices that is being kept
  cube p_array(z.n_rows, z.n_cols , num_samples_kept);
  cube psi_array(z.n_rows, z.n_cols , num_samples_kept);
  //-----------------------------------------------------------------------
  
  int thin_index = 0;
  int sample_index = 0;
  
  //sampling
  //now do the sampling here
  for (int isamples=0; isamples<ndraws; isamples++){
    
    //add in an interuptor. i.e. escape if the user cancels operations
    //checks every 1000 iterations
    if (isamples % 1000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    //Rcpp::Rcout << "\n ------------------------------- " << std::endl;
    
    //sample tau_alpha
    alpha_quad(0) = 0;
    
    for (int i_species = 0; i_species < ns; i_species++){
      alpha_quad += (alpha.col(i_species) - mu_alpha).t()*(alpha.col(i_species) - mu_alpha);
    }
    //loop might be quicker!
    //alpha_quad = accu(square(alpha)) - 2*sum(alpha,1).t()*mu_alpha + ns*mu_alpha.t()*mu_alpha ;
    
    //Rcpp::Rcout << "\n alpha_quad = " << alpha_quad << std::endl;
    //Rcpp::Rcout << "\n  alpha_quad = " <<  alpha_quad << std::endl;
    
    //inv_scale_alpha =  1.0/( inv_nabla_alpha + 0.5*alpha_quad(0) ) ; 
    //tau_alpha = rgammadouble(1, sigma2_alpha_a, inv_scale_alpha );
    
    //scale_alpha =  inv_nabla_alpha + 0.5*alpha_quad(0)  ; 
    //tau_alpha = rgammadouble(1, sigma2_alpha_a, scale_alpha );
    
    tau_alpha = rgammadouble(1, sigma2_alpha_a, inv_nabla_alpha + 0.5*alpha_quad(0) );
    //----------------------------------------------------------------------------
    
    //sample nabla_alpha
    inv_nabla_alpha = rgammadouble(1, 0.5, tau_alpha + inv_A2 );
    //----------------------------------------------------------------------------
    
    //sample tau_beta
    beta_quad(0) = 0;
    for (int i_species = 0; i_species < ns; i_species++){
      beta_quad += (beta.col(i_species) - mu_beta).t()*(beta.col(i_species) - mu_beta);
    }
    //beta_quad = accu(square(beta)) - 2*sum(beta,1).t()*mu_beta + ns*mu_beta.t()*mu_beta ;
    
    //inv_scale_beta =  1.0/( inv_nabla_beta + 0.5*beta_quad(0) ) ; 
    //tau_beta = rgammadouble(1, sigma2_beta_a, inv_scale_beta );
    
    //scale_beta =  inv_nabla_beta + 0.5*beta_quad(0)  ; 
    //tau_beta = rgammadouble(1, sigma2_beta_a, scale_beta );
    
    tau_beta = rgammadouble(1, sigma2_beta_a, inv_nabla_beta + 0.5*beta_quad(0) );
    //----------------------------------------------------------------------------
    
    //sample nabla_beta
    inv_nabla_beta = rgammadouble(1, 0.5, tau_beta + inv_B2 );
    //----------------------------------------------------------------------------
    
    //sample mu_alpha
    //sum(alpha, 1): alpha is a matrix of dimension nd by ns. sum across the rows.
    //I_nd is diagonal matrix of dimension n_d by n_d
    //mu_alpha elements coukd be sampled separately due to independence 
    //Rcpp::Rcout << "\n mu_alpha= \n" << std::endl;
    mu_alpha = mvnrnd( (a2*tau_alpha/(1.0 + ns*a2*tau_alpha))*sum(alpha, 1), (a2/(1.0 + ns*a2*tau_alpha))*I_nd , 1);
    //----------------------------------------------------------------------------
    
    //sample mu_beta
    //Rcpp::Rcout << "\n mu_beta= \n" << std::endl;
    mu_beta = mvnrnd( (b2*tau_beta/(1.0 + ns*b2*tau_beta))*sum(beta, 1), (b2/(1.0 + ns*b2*tau_beta))*I_no , 1);
    //----------------------------------------------------------------------------
    
    //sample beta_i and theta_i
    //this has to be stored in a big matrix!
    for (int i_species = 0; i_species < ns; i_species++){
      
      Xs_beta = X*beta.col(i_species); //arma::mat
      Ks_theta = K*theta.col(i_species); //arma::mat
      
      //sample from Polya-gamm variables in turn for each of the species
      pg_beta = rpg5( Xs_beta + Ks_theta );
      
      //sample beta_i
    //Rcpp::Rcout << "\n beta start " << std::endl;
      cov_beta_i = inv_sympd( tau_beta*I_no + X_t*diagmat( pg_beta )*X ); //arma::mat
      mu_beta_i =  cov_beta_i*(X_t*( z.col(i_species) - 0.5 - diagmat( pg_beta )*( Ks_theta) ) + tau_beta*mu_beta); // arma::mat
      beta.col(i_species) = mvrnormArma2(1, mu_beta_i, cov_beta_i);
    //Rcpp::Rcout << "\n beta end " << std::endl;
      
      //sample theta_i
    //Rcpp::Rcout << "\n theta start " << std::endl;
    //Rcpp::Rcout << "\n tau_i =" << tau_i << std::endl;
    //Rcpp::Rcout << "\n tau_i ...= " << tau_i(0,0) << std::endl;
    //Rcpp::Rcout << "\n tau_i ...= " << tau_i(i_species,0) << std::endl;
    //Rcpp::Rcout << "\n tau_i ...= " << tau_i(i_species-1,0) << std::endl;
    //Rcpp::Rcout << "\n tau_i ...= " << tau_i(i_species-1) << std::endl;
    
      cov_theta_i = inv_sympd( tau_i(i_species)*Minv + K_t*diagmat( pg_beta )*K );
      mu_theta_i = cov_theta_i*K_t*( z.col(i_species) - 0.5 - diagmat( pg_beta )*( Xs_beta  ) );
      theta.col(i_species) = mvrnormArma2(1, mu_theta_i, cov_theta_i);
    //Rcpp::Rcout << "\n beta end " << std::endl;  
    
      //sample tau_i
    //Rcpp::Rcout << "\n tau_i start " << std::endl;
      inv_scale =  theta.col(i_species).t()*Minv*theta.col(i_species)*0.5 + b_tau ; //not inverse scale. rgammadouble uses 1/inv_scale = 1/( theta.t()*Minv*theta *0.5 + i2 );
      tau = rgammadouble(1, a_tau + 0.5*r, inv_scale(0) );
    //Rcpp::Rcout << "\n tau_i end " << std::endl;
      tau_i(i_species) = tau;
    }//end i_species
    
    for ( int i_species = 0; i_species < ns; i_species++){
      
      //find the elements of z associated with z==1
      z_equals1_rows = find(z.col(i_species)==1);  //row number as specified by c++. if an element say 0 ==> row 0 of z is 1.
      V_iter = V.rows(z_equals1_rows);
      pg_alpha  = rpgK5( nsitevisits(z_equals1_rows),  V_iter*alpha.col(i_species) );
      
      //Rcpp::Rcout << "inv(cov_alpha_i) \n" << tau_alpha*I_nd + V_iter.t()*diagmat(pg_alpha)*V_iter  << std::endl;
      cov_alpha_i = inv_sympd( tau_alpha*I_nd + V_iter.t()*diagmat(pg_alpha)*V_iter );
      
      //inefficient
      Y_temp = Y.row(i_species); //viewed as a row vector
      Y_iter = Y_temp.cols(z_equals1_rows);
      
      mu_alpha_i =  cov_alpha_i*(V_iter.t()*( Y_iter.t() - 0.5*nsitevisits(z_equals1_rows) ) + tau_alpha*mu_alpha); // arma::mat
      alpha.col(i_species) = mvrnormArma2(1, mu_alpha_i, cov_alpha_i);
    }
    
    //calculate p and psi
    //use the invlogit function later!
    for (int i_species=0; i_species<ns; i_species++){
      p.col(i_species) = 1.0/(1.0 + exp( -V*alpha.col(i_species)) ); 
      psi.col(i_species) = 1.0/(1.0 + exp( -X*beta.col(i_species)) );
    }
    
    //sample from z
    for (int i_species=0; i_species<ns; i_species++){
      for (int i_sites=0; i_sites<J; i_sites++){
        
        //note Y and z are defined differently
        
        if (Y(i_species, i_sites) == 0){
          prob = 1.0/( 1.0 + (1.0 - psi(i_sites, i_species) )/( psi(i_sites, i_species)*pow(1-p(i_sites, i_species), nsitevisits(i_sites) )) );
          zdraw = rbinom(1,1, prob(0));
          z(i_sites, i_species) = zdraw(0);
        }//endif
      }//end i_sites
    }//end i_species

    //store the samples
    isamples_counter = isamples - num_burnin;
    
    if (isamples_counter>=0){
      post_mu_alpha.col(isamples_counter) = mu_alpha;
      post_mu_beta.col(isamples_counter) = mu_beta;
      
      post_tau_alpha.col(isamples_counter) = tau_alpha;
      post_tau_beta.col(isamples_counter) = tau_beta;
      post_tau_i.col(isamples_counter) = tau_i;
      
      alpha_array.slice(isamples_counter) = alpha;
      beta_array.slice(isamples_counter) = beta;
      z_array.slice(isamples_counter) = z;
      
      p_array.slice(isamples_counter) = p;
      psi_array.slice(isamples_counter) = psi;
    }
    
  }//end sampling
  
  
  return List::create(_["mu_alpha"]=post_mu_alpha,
                      _["mu_beta"]=post_mu_beta,
                      _["tau_alpha"]=post_tau_alpha,
                      _["tau_beta"]=post_tau_beta,
                      _["tau"]=post_tau_i,
                      _["alpha"]=alpha_array,
                      _["beta"]=beta_array,
                      _["detec_probs"]=p_array,
                      _["occ_probs"]=psi_array,
                      _["z"]=z_array);
  
}
