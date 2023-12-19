import numpy as np
import hmmlearn.hmm as hmm

class HMMTrainer(object):
    def __init__(self, model_name = 'GaussianHMM', n_component = 4, cov_type = 'diag', n_iter = 1000) -> None:
        self.model_name = model_name
        self.n_component = n_component
        self.cov_type = cov_type
        self.n_iter = n_iter

        self.models = []
        
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_component, covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')
        
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    def get_score(self, input_data):
        return self.model.score(input_data)
    
    def get_hidden_states(self, input_data):
        return self.model.predict(input_data)
    
    def get_hidden_states_prob(self, input_data):
        return self.model.predict_proba(input_data)
    
    def get_transition_matrix(self):
        return self.model.transmat_
    
    def get_emission_matrix(self):
        return self.model.emissionprob_
    
    def evaluate(self, vaildate_data):
        

if __name__ == "__main__":
    model = HMMTrainer(model_name='GuassianHMM', n_component=10)