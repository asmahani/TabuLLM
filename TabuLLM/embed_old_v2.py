import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai

class TextColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for converting one or more text columns to a numeric matrix using various embedding models.

    This class supports the following text embedding models:
    - OpenAI's large language models (LLM) for embedding
    - Sentence-Transformer (ST) embedding LLMs
    - Doc2Vec (via gensim package)
    - Google Vertex AI's embedding LLMs

    Parameters
    ----------
    model_type : str, default='doc2vec'
        The type of embedding model to use. Must be one of 'openai', 'st', 'doc2vec', or 'google'.
    openai_args : dict, optional
        The arguments for accessing OpenAI's embedding models.
    google_args : dict, optional
        The arguments for accessing Google Vertex AI embedding models.
    st_args : dict, optional
        The arguments for accessing Sentence-Transformer embedding models.
    doc2vec_args : dict, optional
        The arguments for accessing Doc2Vec models.
    return_cols_prefix : str, default='X_'
        The prefix for the returned embedding columns.
    colsep : str, default=' || '
        The column separator for concatenating multiple text columns, if applicable.

    Raises
    ------
    ValueError
        If an invalid model type or insufficient arguments for the specified model are provided.
    TypeError
        If the input columns are not of string type.
    """
    
    def __init__(
        self,
        model_type='doc2vec',
        openai_args=None,
        google_args=None,
        st_args=None,
        doc2vec_args=None,
        colsep=' || ',
        return_cols_prefix='X_'
    ):
        # Assign constructor parameters to instance attributes without modification
        self.model_type = model_type
        self.openai_args = openai_args
        self.google_args = google_args
        self.st_args = st_args
        self.doc2vec_args = doc2vec_args
        self.colsep = colsep
        self.return_cols_prefix = return_cols_prefix

        # Internal state variables (not modifiable after initialization)
        self.doc2vec_fit = None
    
    def _validate_and_prepare_args(self):
        # Validate model_type and set default arguments in fit method
        if self.model_type not in ('openai', 'st', 'doc2vec', 'google'):
            raise ValueError('Invalid model type')
        
        # Modify arguments based on the model type
        if self.model_type == 'openai':
            if self.openai_args is None:
                raise ValueError('OpenAI arguments must be provided for OpenAI model')
            if 'model' not in self.openai_args:
                self.openai_args['model'] = 'text-embedding-3-large'
            if 'client' not in self.openai_args:
                raise ValueError('OpenAI client must be provided for OpenAI model')
        
        if self.model_type == 'google':
            if self.google_args is None:
                raise ValueError('Google arguments must be provided for Google model')
            if 'model' not in self.google_args:
                self.google_args['model'] = 'text-embedding-004'
            if 'task' not in self.google_args:
                self.google_args['task'] = 'SEMANTIC_SIMILARITY'
            if 'batch_size' not in self.google_args:
                self.google_args['batch_size'] = 250
            if 'project_id' not in self.google_args:
                raise ValueError('Google project ID must be provided for Google model')
            if 'location' not in self.google_args:
                raise ValueError('Google location must be provided for Google model')
        
        if self.model_type == 'st':
            if self.st_args is None:
                self.st_args = {}
            if 'model' not in self.st_args:
                self.st_args['model'] = 'sentence-transformers/all-MiniLM-L6-v2'

        if self.model_type == 'doc2vec':
            if self.doc2vec_args is None:
                self.doc2vec_args = {}
            if 'model' not in self.doc2vec_args:
                self.doc2vec_args['model'] = 'PV-DM'
            if 'epochs' not in self.doc2vec_args:
                self.doc2vec_args['epochs'] = 40
            if 'vector_size' not in self.doc2vec_args:
                self.doc2vec_args['vector_size'] = 50
    
    def prep_X(self, X):
        """
        Preprocess the input DataFrame X by concatenating all column values into a single string for each row.
        
        This method converts all columns in the DataFrame to strings, fills any missing values with empty strings, 
        and then concatenates the column values using the specified column separator (`self.colsep`). The resulting 
        list of concatenated strings is returned.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing the text columns to be transformed. All columns must be of string (object) type.

        Returns
        -------
        list of str
            A list of concatenated strings, where each string corresponds to a row in the DataFrame with the column values
            separated by `self.colsep`.
        
        Raises
        ------
        TypeError
            If any column in the input DataFrame is not of string (object) type.

        Example
        -------
        >>> df = pd.DataFrame({'col1': ['hello', 'world'], 'col2': ['foo', 'bar']})
        >>> transformer = TextColumnTransformer(colsep=' || ')
        >>> transformer.prep_X(df)
        ['col1: hello || col2: foo', 'col1: world || col2: bar']
        """
        if not (X.dtypes == 'object').all():
            raise TypeError('All columns of X must be of string (object) type')
        
        Xstr = X.fillna('').astype(str).apply(
            lambda row: self.colsep.join([f'{col}: {row[col]}' for col in X.columns]), axis=1
        ).tolist()
        return Xstr

    def _fit_doc2vec(self, X, y=None):
        """
        Fit a Doc2Vec model on the preprocessed text data.

        This method preprocesses the input DataFrame using `prep_X()` to concatenate text from all columns. 
        It then constructs a corpus using the preprocessed text and fits a Doc2Vec model based on the parameters 
        specified in `self.doc2vec_args`. The fitted Doc2Vec model is stored in `self.doc2vec_fit`.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame containing text data to be transformed into embeddings.
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            The instance itself with the fitted Doc2Vec model stored in `self.doc2vec_fit`.

        Raises
        ------
        ValueError
            If an invalid `model` type is provided in `self.doc2vec_args`.
        """
        Xstr = self.prep_X(X)
        args = self.doc2vec_args
        corpus = [TaggedDocument(words=simple_preprocess(doc), tags=[str(i)]) for i, doc in enumerate(Xstr)]
        alg = 1 if args['model'] == 'PV-DM' else 0
        model = Doc2Vec(corpus, vector_size=args['vector_size'], window=2, min_count=1, epochs=args['epochs'], dm=alg)
        self.doc2vec_fit = model
        return self
    
    def fit(self, X, y=None):
        """
        Fit the transformer on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit.
        y : array-like of shape (n_samples,), optional
            The target values (ignored).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Perform argument validation and preparation
        self._validate_and_prepare_args()

        if self.model_type == 'doc2vec':
            return self._fit_doc2vec(X, y)
        
        return self

    def fit_transform(self, X, y=None):
        """
        Fit the transformer on the data and then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit and transform.
        y : array-like of shape (n_samples,), optional
            The target values (ignored).

        Returns
        -------
        embeddings : numpy.ndarray of shape (n_samples, embedding_dim)
            The transformed embeddings.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X):
        """
        Transform the data into embeddings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        embeddings : numpy.ndarray of shape (n_samples, embedding_dim)
            The transformed embeddings.
        """
        Xstr = self.prep_X(X)

        if self.model_type == 'openai':
            arr = self._transform_openai(Xstr)
        elif self.model_type == 'st':
            arr = self._transform_st(Xstr)
        elif self.model_type == 'doc2vec':
            arr = self._transform_doc2vec(Xstr)
        elif self.model_type == 'google':
            arr = self._transform_google(Xstr)
        else:
            raise ValueError('Invalid model type')
        
        return pd.DataFrame(arr, columns=[self.return_cols_prefix + str(i) for i in range(arr.shape[1])])

    def _transform_google(self, X):
        """
        Transform each element of the input text list using Google Vertex AI's emebedding models.

        Parameters
        ----------
        X : list of str
            List of texts to be embedded into numeric vectors.
        
        Returns
        -------
        embeddings : numpy.ndarray
            A 2D array, each row being an embedding vector corresponding to an element of X.
        """
        args = self.google_args
        vertexai.init(project=args['project_id'], location=args['location'])
        model = TextEmbeddingModel.from_pretrained(args['model'])
        inputs = [TextEmbeddingInput(text, args['task']) for text in X]
        embeddings = []
        batch_size = args['batch_size']
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_embeddings = model.get_embeddings(batch_inputs)
            embeddings.extend(batch_embeddings)
        return np.array([embedding.values for embedding in embeddings])

    def _transform_doc2vec(self, X):
        """
        Transform each element of the input text list using a fitted Doc2Vec embedding model.

        Parameters
        ----------
        X : list of str
            List of texts to be embedded into numeric vectors.
        
        Returns
        -------
        embeddings : numpy.ndarray
            A 2D array, each row being an embedding vector corresponding to an element of X.
        """
        return np.array([self.doc2vec_fit.infer_vector(simple_preprocess(doc)) for doc in X])

    def _transform_st(self, X):
        """
        Transform each element of the input text list using a sentence transformer model.

        Parameters
        ----------
        X : list of str
            List of texts to be embedded into numeric vectors.
        
        Returns
        -------
        embeddings : numpy.ndarray
            A 2D array, each row being an embedding vector corresponding to an element of X.
        """
        model = SentenceTransformer(self.st_args['model'])
        return model.encode(X)

    def _transform_openai(self, X):
        """
        Transform each element of the input text list using an OpenAI embedding model.

        Parameters
        ----------
        X : list of str
            List of texts to be embedded into numeric vectors.
        
        Returns
        -------
        embeddings : numpy.ndarray
            A 2D array, each row being an embedding vector corresponding to an element of X.
        """
        args = self.openai_args
        ret = args['client'].embeddings.create(input=X, model=args['model'])
        return np.array([ret.data[n].embedding for n in range(len(ret.data))])
