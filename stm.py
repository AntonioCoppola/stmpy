# Set up the environment
import numpy as np
import scipy as sp
from scipy import io
from scipy import optimize

import time
import os

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import warnings
warnings.filterwarnings('ignore')

# R Imports for STM
for f in os.listdir("R"):
    if f not in ['.DS_Store', '.Rapp.history']:
        robjects.r.source("R/" + f)

# R library imports
for pkg in ['Matrix', 'stringr', 'splines', 'matrixStats', 
    'slam', 'lda', 'glmnet', 'magrittr']:
    try:
        robjects.r("library(" + pkg + ")")
    except:
        raise Exception("Missing R library " + pkg + 
            ". Please ensure the R requirements are satisfied.")

# Core class definition
class STM:

    def __init__(self, sc):
        self.sc = sc
        self.status = 0

    # Conversion function: R list to Python dict
    def __rlist_2py__(self, rlist):
        return dict(zip(rlist.names,
                   list(rlist)))

    # Conversion function: beta_ss
    def __pybeta_ss_2r__(self, beta):
        return robjects.ListVector( 
            robjects.ListVector({str(i+1):robjects.Matrix(mat) for i, mat in enumerate(beta)}))

    # Conversion function: beta
    def __pybeta_2r__(self, beta):
        return robjects.ListVector({'beta':
                         robjects.ListVector({str(i+1):robjects.Matrix(mat) for i, mat in enumerate(beta)})
                        })

    # Conversion function: mu
    def __rmu_2py__(self, mu):
        out = {'mu': np.array(mu[0])}
        if len(mu) > 1:
            out['gamma'] = np.array(mu[1])
        else:
            out['gamma'] = None
        return out

    # Likelihood
    def likelihood(self, eta, beta, doc_ct, mu, siginv):
        exp_eta = np.exp(np.append(eta, np.array([0])))
        ndoc = np.sum(doc_ct)
        part1 = np.dot(np.log(np.dot(exp_eta, beta)), doc_ct) - ndoc * np.log(np.sum(exp_eta))
        diff = mu.T - eta
        part2 = 0.5 * float(np.dot(np.dot(diff, siginv), diff.T))
        return part2 - part1

    # Gradient
    def grad(self, eta, beta, doc_ct, mu, siginv):
        exp_eta = np.exp(np.append(eta, [0]))
        beta_prime = np.apply_along_axis(lambda x: x * exp_eta, 0, beta)
        part1 = np.dot(beta_prime, doc_ct/np.sum(beta_prime, 0).T) - (np.sum(doc_ct)/ np.sum(exp_eta)) * exp_eta
        diff = mu.T - eta
        part2 = np.dot(siginv, diff.T)
        part1 = part1[:len(part1)-1]
        return (part2.T - part1).flatten()

    # Hessian - Phi - Bound
    def hpb(self, eta, beta, doc_ct, mu, siginv, sigmaentropy):
        
        # Compute the Hessian
        exp_eta = np.exp(np.append(eta, [0]))
        theta = np.reshape(exp_eta/np.sum(exp_eta), (len(exp_eta), -1)).T
        EB = np.apply_along_axis(lambda x: x * exp_eta, 0, beta)
        EB = np.apply_along_axis(lambda x: x * (np.sqrt(doc_ct).T) / np.sum(EB,0), 1, EB)
        hess = np.dot(EB, EB.T) - np.sum(doc_ct) * np.dot(theta.T, theta)    
        EB = np.apply_along_axis(lambda x: x * np.sqrt(doc_ct).T, 1, EB)
        hess[np.diag_indices_from(hess)] = hess[np.diag_indices_from(hess)] - np.sum(EB, 1) + np.sum(doc_ct) * theta
        hess = hess[:hess.shape[0]-1,:hess.shape[1]-1] + siginv

        # Invert via Cholesky decomposition
        try:
            nu = np.linalg.cholesky(hess)
        except:
            dvec = np.array(np.diag(hess))
            magnitudes = np.sum(np.abs(hess), 1) - abs(dvec)
            Km1 = len(dvec)
            for i in range(Km1):
                if dvec[i] < magnitudes[i]:
                    dvec[i] = magnitudes[i]
            hess[np.diag_indices_from(hess)] = dvec
            nu = np.linalg.cholesky(hess)

        # Finish construction
        det_term = -np.sum(np.log(np.diag(nu)))
        nu = np.linalg.inv(np.triu(nu))
        nu = np.dot(nu, nu.T)
        diff = eta - mu.flatten()

        # Compute the bound
        bound = (np.dot(np.log(np.dot(theta, beta)), doc_ct) + det_term 
                 - 0.5 * np.dot(diff.T, np.dot(siginv, diff)) - sigmaentropy)

        # Construct output
        out = {'phis': EB,
               'eta': {'lambda': eta, 'nu': nu},
               'bound': bound}
        return out

    # Code for worker node dispatch
    def estep_docloop(self, doc_item, siginv, sigmaentropy):
        
        # Extract the info
        doc_ct = doc_item['doc'][1]
        eta = doc_item['init']
        beta = doc_item['beta_i']
        mu = doc_item['mu_i']
        
        # Run the step
        try:
            optim_par = sp.optimize.minimize(likelihood, eta, args=(beta, doc_ct, mu, siginv), 
                                    method='BFGS')
            out = hpb(optim_par.x, beta, doc_ct, mu, siginv, sigmaentropy)
        except:
            import scipy as sp
            import numpy as np
            from scipy import optimize
            optim_par = sp.optimize.minimize(likelihood, eta, args=(beta, doc_ct, mu, siginv), 
                                    method='BFGS')
            out = hpb(optim_par.x, beta, doc_ct, mu, siginv, sigmaentropy)
        
        # Also include aspect and  in the output
        out['aspect'] = doc_item['aspect']
        out['words'] = doc_item['doc'][0]
        
        return out


    # Estep on Spark
    def estep_spark(self, documents, beta_index, beta, Lambda_old,
                    mu, sigma, verbose, sc, update_mu=False):
        
        # Initialize sufficient statistics
        sigma_ss = np.zeros((K-1, K-1))
        beta_ss = [np.zeros((K, V)) for i in range(A)]
        Lambda = np.zeros((N, N))
        siginv = np.linalg.inv(sigma)
        sigmaentropy = np.log(np.abs(np.linalg.det(sigma))) * 0.5
        
        # Parallelize document collection
        collection = [{'doc':doc, 'aspect': int(aspect), 'init': init} 
                      for (doc, aspect, init) in zip(documents, beta_index, Lambda_old)]
        for i, item in enumerate(collection):
            item['beta_i'] = beta[item['aspect']-1][:,[x-1 for x in item['doc'][0]]]
            if mu['gamma'] is None:
                item['mu_i'] = mu['mu']
            else:
                item['mu_i'] = mu['mu'][:,i]
        
        # Run estep on Spark
        collection_par = sc.parallelize(collection)
        doc_results = collection_par.map(lambda x: estep_docloop(x, siginv, sigmaentropy)).collect()
        
        # Update sufficient statistics
        sigma_ss += reduce(lambda a, b: a + b, [doc['eta']['nu'] for doc in doc_results])
        bound = np.array([doc['bound'][0] for doc in doc_results])
        Lambda = np.array([doc['eta']['lambda'] for doc in doc_results])
        
        # Update beta
        for doc in doc_results:
            beta_ss[doc['aspect']-1][:,[j-1 for j in doc['words']]] -= doc['phis']
        
        return sigma_ss, beta_ss, bound, Lambda

    def train(self, data, document_col, K, prevalence, content, init_type="Spectral", 
        seed=None, max_em_its=500, emtol=1e-5, verbose=True, reportevery=5, 
        LDAbeta=True, interactions=True, ngroups=1, model=None, gamma_prior="Pooled", 
        sigma_prior=0, kappa_prior="L1"):

        # Blocked updates are not yet supported
        if ngroups > 1:
            raise Exception("Blocked inference not yet supported. ")

        # Push the data to R
        pandas2ri.activate()
        try:
            robjects.globalenv['doc_df'] = pandas2ri.py2ri(cond_mat_mc)
        except:
            raise Exception("The data must be in the form of a Pandas dataframe")

        # Parse any date objects
        robjects.r('''
            if("date" %in% colnames(doc_df)){
                doc_df$date = as.Date(doc_df$date)
            }
        ''');

        # Prep the corpus
        robject.r("processed_corpus_temp = textProcessor(doc_df$" + document_col + 
            ", metadata=doc_df, lowercase=TRUE)");
        robjects.r('''
            processed_corpus = prepDocuments(processed_corpus_temp$documents,
                                         processed_corpus_temp$vocab, 
                                         processed_corpus_temp$meta,
                                         lower.thresh=1)
            rm(processed_corpus_temp); invisible(gc())
        ''');

        # Initialize
        robjects.r('''
            fit = stm(processed_corpus$documents, 
                     processed_corpus$vocab,
                     K=20, prevalence=~s(date),
                     data=processed_corpus$meta,
                     init.type = 'Spectral',
                     seed=02138)
        ''');

        # Finish initialization
        robjects.r('''

            # Extract objects
            documents <- fit$documents
            vocab <- fit$vocab
            settings <- fit$settings 
            model <- fit$model
            verbose <- settings$verbose
            
            # Initialize parameters
            ngroups <- settings$ngroups
            if(is.null(model)) {
            
                if(verbose) cat("Beginning Initialization.\n")
                model <- stm.init(documents, settings)
                
                # If using the Lee and Mimno method of setting K, update the settings
                if(settings$dim$K==0) settings$dim$K <- nrow(model$beta[[1]])
                
                # Unpack
                mu <- list(mu=model$mu)
                sigma <- model$sigma
                beta <- list(beta=model$beta)
                if(!is.null(model$kappa)) beta$kappa <- model$kappa
                lambda <- model$lambda
                convergence <- NULL 
                
                # Discard the old object
                # rm(model)
            } else {
            
                if(verbose) cat("Restarting Model...\n")
                
                # Extract from a standard STM object so we can simply continue
                mu <- model$mu
                beta <- list(beta=lapply(model$beta$logbeta, exp))
                if(!is.null(model$beta$kappa)) beta$kappa <- model$beta$kappa
                sigma <- model$sigma
                lambda <- model$eta
                convergence <- model$convergence
                
                # Manually declare the model not converged or it will stop after the first iteration
                convergence$stopits <- FALSE
                convergence$converged <- FALSE
                
                # Iterate by 1 as that would have happened otherwise
                convergence$its <- convergence$its + 1 
            }    
          
            #Pull out some book keeping elements
            ntokens <- sum(settings$dim$wcounts$x)
            betaindex <- settings$covariates$betaindex
            stopits <- FALSE
            if(ngroups!=1) {
                groups <- cut(1:length(documents), breaks=ngroups, labels=FALSE) 
            }
            suffstats <- vector(mode="list", length=ngroups)
        ''');

        # Get the settings
        fit = dict(zip(robjects.globalenv['fit'].names, 
                 list(robjects.globalenv['fit'])))
        settings = dict(zip( fit['settings'].names, 
                 list(fit['settings'])))
        K, A, V, N = [int(settings['dim'][i][0]) for i in range(4)]


        # Some setup for EM, retrieving the R objects
        stopits = False
        ngroups = int(robjects.globalenv['ngroups'][0])
        documents = [np.array(x) for x in list(robjects.globalenv['documents'])]
        beta_index = np.array(robjects.globalenv['betaindex'])
        beta = [np.array(x) for x in robjects.globalenv['beta'][0]]
        Lambda = np.array(robjects.globalenv['lambda'])
        mu = self.__rmu_2py__(robjects.globalenv['mu'])
        sigma = np.array(robjects.globalenv['sigma'])
        verbose = settings['verbose'][0]


        # Clear the convergence object
        try:
            del convergence
        except:
            pass

        # Initiate counter
        i = 0

        # Run EM!
        while stopits is not True:
            
            # Non-blocked updates
            if ngroups==1:
                
                # Run the E-step
                sigma_ss, beta_ss, bound_ss, Lambda = estep_spark(documents, beta_index, beta, 
                                                                  Lambda, mu, sigma, verbose, sc, False)
                if verbose:
                    print "Completed E-Step, Iteration " +  str(i+1)
                
                # Run the M-step
                mu = self.__rmu_2py__(robjects.globalenv['opt.mu'](Lambda, 
                                     robjects.r("settings$gamma$mode"),
                                     robjects.r("settings$covariates$X"),
                                     robjects.r("settings$gamma$enet")))
                sigma = np.array(robjects.r("opt.sigma")(sigma_ss, 
                                                         Lambda, 
                                                         mu['mu'], 
                                                         float(robjects.r("settings$sigma$prior"))))
                beta_new_temp = robjects.globalenv['opt.beta'](self.__pybeta_ss_2r__(beta_ss), 
                                                               robjects.r('beta$kappa'), 
                                                               robjects.r('settings'))
                beta_new = [np.array(x) for x in beta_new_temp[0]]
                
                if verbose:
                    print "Completed M-Step, Iteration " +  str(i+1)
            
            # Check for convergence
            try:
                convergence = robjects.globalenv['convergence.check'](bound_ss, convergence, 
                                                                      robjects.globalenv['settings'])
            except:
                convergence = robjects.globalenv['convergence.check'](bound_ss, 
                                                                      robjects.globalenv['convergence'], 
                                                                      robjects.globalenv['settings'])
                
            # Print updates if we have not converged
            stopits = bool(self.__rlist_2py__(convergence)['stopits'][0])
            if stopits is not True and verbose:
                robjects.r.report(convergence, robjects.r.ntokens, self.__pybeta_2r__(beta),
                         robjects.r.vocab, robjects.r('settings$topicreportevery'),
                         verbose);
            
            # Increase counter
            i += 1
            print ""
