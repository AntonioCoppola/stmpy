# Set up the environment
import rpy2
import time
import os
import warnings
import pkgutil
import numpy as np
import scipy as sp
import rpy2.robjects as robjects
from scipy import io
from scipy import optimize
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
warnings.filterwarnings('ignore')

# Intall any missing R packages
rpy2.robjects.r('''
    packages <- c('Matrix', 'stringr', 'splines', 'matrixStats', 
    'slam', 'lda', 'glmnet', 'magrittr', 'tm', 'SnowballC')

    if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
        cat("Installation of certain R packages is required for stmpy\n")
        install.packages(setdiff(packages, rownames(installed.packages())), 
            repos='http://cran.us.r-project.org')  
    }
''')

# R imports for STM
for __f__ in ['checkBeta.R', 'checkFactors.R', 'cloud.R', 'dmr.R', 
    'estimateEffect.R', 'exclusivity.R', 'findThoughts.R', 'findTopic.R', 
    'heldout.R', 'jeffreyskappa.R', 'labelTopics.R', 'manyTopics.R', 
    'multiSTM.R', 'parseFormulas.R', 'permute.R', 'plot.estimateEffect.R', 
    'plot.searchK.R', 'plot.STM.R', 'plotModels.R', 'plotQuote.R', 
    'plotRemoved.R', 'plottingutilfns.R', 'plotTopicLoess.R', 'prepDocuments.R',
    'produce_cmatrix.R', 'RcppExports.R', 'readCorpus.R', 'residuals.R', 's.R',
    'sageLabels.R', 'searchK.R', 'selectModel.R', 'semanticCoherence.R', 
    'simBetas.R', 'spectral.R', 'stm.control.R', 'stm.R', 'STMconvergence.R', 
    'STMestep.R', 'STMfunctions.R', 'STMinit.R', 'STMlncpp.R', 'STMmnreg.R', 
    'STMmu.R', 'STMoptbeta.R', 'STMreport.R', 'STMsigma.R', 'summary.STM.R', 
    'tau.R', 'textProcessor.R', 'thetaPosterior.R', 'toLDAvis.R', 'topicCorr.R', 
    'topicLasso.R', 'topicQuality.R', 'writeLdac.R']:
    data = pkgutil.get_data('stmpy', 'R/' + __f__)
    robjects.r(data)

# R library imports
for __pkg__ in ['Matrix', 'stringr', 'splines', 'matrixStats', 
    'slam', 'lda', 'glmnet', 'magrittr']:
    try:
        robjects.r("library(" + __pkg__ + ")")
    except:
        raise Exception("Missing R library " + __pkg__ + 
            ". Please ensure the R requirements are satisfied.")

# Likelihood for workers
def __likelihood__(eta, beta, doc_ct, mu, siginv):
    exp_eta = np.exp(np.append(eta, np.array([0])))
    ndoc = np.sum(doc_ct)
    part1 = np.dot(np.log(np.dot(exp_eta, beta)), doc_ct) - ndoc * np.log(np.sum(exp_eta))
    diff = mu.T - eta
    part2 = 0.5 * float(np.dot(np.dot(diff, siginv), diff.T))
    return part2 - part1

# Gradient for workers
def __grad__(eta, beta, doc_ct, mu, siginv):
    exp_eta = np.exp(np.append(eta, [0]))
    beta_prime = np.apply_along_axis(lambda x: x * exp_eta, 0, beta)
    part1 = np.dot(beta_prime, doc_ct/np.sum(beta_prime, 0).T) - (np.sum(doc_ct)/ np.sum(exp_eta)) * exp_eta
    diff = mu.T - eta
    part2 = np.dot(siginv, diff.T)
    part1 = part1[:len(part1)-1]
    return (part2.T - part1).flatten()

# Hessian - Phi - Bound
def __hpb__(eta, beta, doc_ct, mu, siginv, sigmaentropy):
    
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
    bound = (np.dot(np.log(np.dot(theta, beta)), doc_ct) + det_term - 0.5 * np.dot(diff.T, np.dot(siginv, diff)) - sigmaentropy)

    # Construct output
    out = {'phis': EB, 'eta': {'lambda': eta, 'nu': nu},'bound': bound}
    return out

# Code for worker node dispatch
def __estep_docloop__(doc_item, siginv, sigmaentropy):
    
    # Extract the info
    doc_ct = doc_item['doc'][1]
    eta = doc_item['init']
    beta = doc_item['beta_i']
    mu = doc_item['mu_i']
    
    # Run the step
    try:
        optim_par = sp.optimize.minimize(__likelihood__, eta, 
            args=(beta, doc_ct, mu, siginv), method='BFGS')
        out = __hpb__(optim_par.x, beta, doc_ct, mu, siginv, sigmaentropy)
    except:
        import scipy as sp
        import numpy as np
        from scipy import optimize
        optim_par = sp.optimize.minimize(__likelihood__, eta, 
            args=(beta, doc_ct, mu, siginv), method='BFGS')
        out = __hpb__(optim_par.x, beta, doc_ct, mu, siginv, sigmaentropy)
    
    # Also include aspect and  in the output
    out['aspect'] = doc_item['aspect']
    out['words'] = doc_item['doc'][0]
    
    return out

# Core class definition
class STM:

    def __init__(self, sc):
        self.sc = sc
        self.trained = False
        self.mu = None
        self.sigma = None
        self.beta = None
        self.settings = None
        self.vocab = None
        self.convergence = None
        self.theta = None
        self.eta = None
        self.invsigma = None
        self.em_time = None

    def __repr__(self):
        return "An object of class STM. Not yet trained."

    def __print__(self):
        return "An object of class STM. Not yet trained."

    # Conversion function: R list to Python dict
    def __rlist_2py__(self, rlist):
        return dict(zip(rlist.names,
                   list(rlist)))
        
    # Conversion function: beta_ss
    def __pybeta_ss_2r__(self, beta):
        out = {}
        for i, mat in enumerate(beta):
            out[str(i+1)] = robjects.Matrix(mat)
        return robjects.ListVector(out)

    # Conversion function: beta
    def __pybeta_2r__(self, beta):
        out = {}
        for i, mat in enumerate(beta):
            out[str(i+1)] = robjects.Matrix(mat)
        return robjects.ListVector({'beta': robjects.ListVector(out)})

    # Conversion function: mu
    def __rmu_2py__(self, mu):
        out = {'mu': np.array(mu[0])}
        if len(mu) > 1:
            out['gamma'] = np.array(mu[1])
        else:
            out['gamma'] = None
        return out

    # Estep on Spark
    def __estep_spark__(self, documents, beta_index, beta, Lambda_old,
                    mu, sigma, verbose, update_mu=False):
        
        # Initialize sufficient statistics
        sigma_ss = np.zeros((self.K-1, self.K-1))
        beta_ss = [np.zeros((self.K, self.V)) for i in range(self.A)]
        Lambda = np.zeros((self.N, self.N))
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
        collection_par = self.sc.parallelize(collection)
        doc_results = collection_par.map(lambda x: __estep_docloop__(x, siginv, sigmaentropy)).collect()
        
        # Update sufficient statistics
        sigma_ss += reduce(lambda a, b: a + b, [doc['eta']['nu'] for doc in doc_results])
        bound = np.array([doc['bound'][0] for doc in doc_results])
        Lambda = np.array([doc['eta']['lambda'] for doc in doc_results])
        
        # Update beta
        for doc in doc_results:
            beta_ss[doc['aspect']-1][:,[j-1 for j in doc['words']]] -= doc['phis']
        
        return sigma_ss, beta_ss, bound, Lambda

    # Function to package the model
    def __package_model__(self):

        model = {"mu": robjects.ListVector(self.mu), 
        "sigma": self.sigma, 
        "beta": robjects.ListVector({"beta": self.logbeta, "logbeta": self.logbeta}),
        "settings": robjects.ListVector(self.settings),
        "vocab": self.vocab,
        "convergence": self.convergence,
        "theta": self.theta,
        "eta": self.eta,
        "invsigma": self.invsigma}

        return robjects.ListVector(model)

    def train(self, data, document_col, K, prevalence=None, content=None, init_type="Spectral", 
        seed=None, max_em_its=500, emtol=1e-5, verbose=True, reportevery=5, 
        LDAbeta=True, interactions=True, ngroups=1, gamma_prior="Pooled", 
        sigma_prior=0, kappa_prior="L1"):

        # Blocked updates are not yet supported
        if ngroups > 1:
            raise Exception("Blocked inference not yet supported. ")

        # Push the data to R
        pandas2ri.activate()
        try:
            robjects.globalenv['doc_df'] = pandas2ri.py2ri(data)
        except:
            raise Exception("The data must be in the form of a Pandas dataframe")

        # Parse any date objects
        robjects.r('''
            if("date" %in% colnames(doc_df)){
                doc_df$date = as.Date(doc_df$date)
            }
        ''');

        # Prep the corpus
        print "Beginning document preprocessing."
        robjects.r("processed_corpus_temp = textProcessor(doc_df$" + document_col + 
            ", metadata=doc_df, lowercase=TRUE)");
        robjects.r('''
            processed_corpus = prepDocuments(processed_corpus_temp$documents,
                                         processed_corpus_temp$vocab, 
                                         processed_corpus_temp$meta,
                                         lower.thresh=1)
            rm(processed_corpus_temp); invisible(gc())
        ''');
        print ""

        # Seed parsing
        yield_seed = lambda x: robjects.NULL if x==None else x

        # Initialize the run
        if prevalence != None and content != None:
            robjects.globalenv["fit"] = robjects.r.stm(
                robjects.r("processed_corpus$documents"), 
                robjects.r("processed_corpus$vocab"), 
                K=K, prevalence=robjects.Formula(prevalence),
                content=robjects.Formula(content),
                data=robjects.r("processed_corpus$meta"),
                init_type=init_type, seed=yield_seed(seed),
                max_em_its=max_em_its, emtol=emtol,
                verbose=verbose, reportevery=reportevery,
                LDAbeta=LDAbeta, interactions=interactions,
                ngroups=ngroups, gamma_prior=gamma_prior,
                sigma_prior=sigma_prior, kappa_prior=kappa_prior)
        elif prevalence != None and content == None:
            robjects.globalenv["fit"] = robjects.r.stm(
                robjects.r("processed_corpus$documents"), 
                robjects.r("processed_corpus$vocab"), 
                K=K, prevalence=robjects.Formula(prevalence),
                data=robjects.r("processed_corpus$meta"),
                init_type=init_type, seed=yield_seed(seed),
                max_em_its=max_em_its, emtol=emtol,
                verbose=verbose, reportevery=reportevery,
                LDAbeta=LDAbeta, interactions=interactions,
                ngroups=ngroups, gamma_prior=gamma_prior,
                sigma_prior=sigma_prior, kappa_prior=kappa_prior)
        elif prevalence == None and content != None:
            robjects.globalenv["fit"] = robjects.r.stm(
                robjects.r("processed_corpus$documents"), 
                robjects.r("processed_corpus$vocab"), 
                K=K, content=robjects.Formula(content),
                data=robjects.r("processed_corpus$meta"),
                init_type=init_type, seed=yield_seed(seed),
                max_em_its=max_em_its, emtol=emtol,
                verbose=verbose, reportevery=reportevery,
                LDAbeta=LDAbeta, interactions=interactions,
                ngroups=ngroups, gamma_prior=gamma_prior,
                sigma_prior=sigma_prior, kappa_prior=kappa_prior)
        else:
            robjects.globalenv["fit"] = robjects.r.stm(
                robjects.r("processed_corpus$documents"), 
                robjects.r("processed_corpus$vocab"), 
                K=K, data=robjects.r("processed_corpus$meta"),
                init_type=init_type, seed=yield_seed(seed),
                max_em_its=max_em_its, emtol=emtol,
                verbose=verbose, reportevery=reportevery,
                LDAbeta=LDAbeta, interactions=interactions,
                ngroups=ngroups, gamma_prior=gamma_prior,
                sigma_prior=sigma_prior, kappa_prior=kappa_prior)

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
        self.K, self.A, self.V, self.N = [int(settings['dim'][i][0]) for i in range(4)]

        # Some setup for EM, retrieving the R objects
        stopits = False
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
            i = 0
        except:
            i = 0

        # Initiate timer
        start_time = time.clock()

        # Run EM!
        while stopits is not True:
            
            # Non-blocked updates
            if ngroups==1:
                
                # Run the E-step
                sigma_ss, beta_ss, bound_ss, Lambda = self.__estep_spark__(documents, beta_index, beta, 
                                                                  Lambda, mu, sigma, verbose, False)
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
                beta = [np.array(x) for x in beta_new_temp[0]]
                
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
            if verbose:
                print ""

        # Some book-keeping
        self.em_time = time.clock() - start_time
        self.trained = True

        # Save the results
        self.beta = beta
        self.logbeta = [np.log(x) for x in self.beta]
        self.Lambda = Lambda
        self.mu = mu
        self.sigma = sigma
        self.settings = settings
        self.vocab = robjects.r.vocab
        self.convergence = convergence
        self.theta = np.array(robjects.r("theta_from_lambda")(Lambda))
        self.eta = np.array(robjects.r("eta_from_lambda")(Lambda))
        self.invsigma = np.linalg.inv(sigma)

    def label_topics(self, n=7, topics=None, frexweight=.5):
        return robjects.r.labelTopics(self.__package_model__(), 
            n=n, frexweight=frexweight)

    def print_topics(self, n=7, topics=None, frexweight=.5):
        robjects.r("print.labelTopics")(self.label_topics(n=n, 
            frexweight=frexweight))

    def find_thoughts(self, texts, topics=1, n=2, thresh=0.0):
        return robjects.r("findThoughts")(self.__package_model__(), texts, 
            n=n, topics=topics)

    def print_thoughts(self, texts, topics=1, n=2, thresh=0.0):
        robjects.r("plot.findThoughts")(self.find_thoughts(texts), n=n,
            topics=topics)

    def estimate_effect(self):
        return None

    def plot_estimate_effect(self):
        return None
