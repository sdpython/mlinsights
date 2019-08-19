"""
@file
@brief Overloads :epkg:`TfidfVectorizer` and :epkg:`CountVectorizer`.
"""
from sklearn.base import TransformerMixin, BaseEstimator


class BaseReciprocalTransformer(BaseEstimator, TransformerMixin):
    """
    Base for transform which transforms the features
    and the targets at the same time. It must also
    return 
    """
    
    def get_fct_inv(self):
        """
        Returns a trained transform which reverse the target
        after a predictor.
        """
        raise NotImplementedError("This must be overwritten.")
    
    def transform(self, X, y):
        """
        Transforms *X* and *y*.
        Returns transformed *X* and *y*.
        """        
        raise NotImplementedError("This must be overwritten.")
