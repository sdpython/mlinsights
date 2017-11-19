"""
@file
@brief Implements a way to get close examples based
on the output of a machine learned model.
"""

class SearchEngineExamples:
    """
    Implements a kind of local search engine which
    looks for similar results based on the output
    of a function such as the predictions of the machine
    leanrned model.
    """
    
    def __init__(self, data=None, index=None, features=None, metadata=None):
        """
        Every observation is described with an id (or index),
        a list of features, a list of metadata.
        
        @param      data        a dataframe or None if the
                                the features and the metadata
                                are specified with an array and a
                                dictionary
        @param      index       column name if data is not None,
                                *range(0, len(data) or len(features))*
                                if None
        @param      features    features columns  or
                                or an array
        @param      metadata    data
        """
        if data is None:
            if not isinstance(features, numpy.ndarray):
                raise TypeError("features must be an array if data is None")
            self.features = features
            self.metadata = metadata
            if index is None:
                self.index = list(range(features.shape[0]))
            else:
                self.index = index
                if max(index) + 1 != self.features.shape[0]:
                    raise ValueError("dimension mismatch {0} != {1}".format(
                            max(index) + 1, self.features.shape[0]))
        else:
            if not isinstance(data, pandas.dataFrame):
                raise ValueError("data should be a dataframe")
            self.index = list(range(data.shape[0]))
            self.features = data.loc[index, features]
            self.metadata = data.loc[index, metadata] if metadata else None
    
    def _build_neighborhood(self):
        pass
        
    def _first_pass(self, X):
        pass
        
    def _second_pass(self, X, first):
        # no second pass
        return first
        
    def searchn(self, obs, n=5):
        """
        Search for neighbors based on *obs*.
        
        @param      obs     observation
        @param      
        @return             list of desired neighbors
        """
        res = self._first_pass(obs)
        res = self._second_pass(obs, res)
        return res
            
     
