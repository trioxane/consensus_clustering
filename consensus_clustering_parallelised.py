import numpy as np
import os
from pathlib import Path
import gc
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

class ConsensusClustering:
    """
    Consensus Clustering algorithm implementaion.
    
    References:
        [1] https://github.com/MrinalJain17/drake/blob/master/drake/cluster.py
        [2] https://en.wikipedia.org/wiki/Consensus_clustering
        [3] https://joblib.readthedocs.io/en/latest/auto_examples/parallel_memmap.html#sphx-glr-auto-examples-parallel-memmap-py
    """

    def __init__(
        self,
        clusterer=None,
        clusterer_options={'n_init': 3},
        K_range=(2, 3),
        n_iterations=25,
        subsampling=0.8,
        random_state=None,
        consensus_matrix_analysis='PAC',
        PAC_interval=(0.1, 0.9),
        plot_cdf=True,
        agg_clustering_linkage='average',
        n_jobs=1,
        parallelization_method='multithreading',
        memmap_folder = './memmap',
    ):
        """
        Initialize ConsensusClustering object.

        Parameters:
        -----------
        clusterer : object, default=None
            The clustering algorithm to use. Default is None, which sets KMeans as the default clusterer.
        clusterer_options : dict, default={'n_init': 3}
            Dictionary of options to be passed to the clusterer.
        K_range : tuple, default=(2,)
            Tuple indicating the range of number of clusters to consider.
        n_iterations : int, default=25
            Number of iterations for subsampling.
        subsampling : float, default=0.8
            Percentage of samples to use for each iteration.
        random_state : int or None, default=None
            Random seed for reproducibility.
        consensus_matrix_analysis : str, default='PAC'
            Method for consensus matrix analysis ('PAC' or 'other').
        PAC_interval : tuple, default=(0.1, 0.9)
            Interval for the PAC analysis.
        plot_cdf : bool, default=True
            Whether to plot the Cumulative Distribution Function (CDF) of the consensus indices.
        agg_clustering_linkage : str, default='average'
            Linkage method for agglomerative clustering.
        n_jobs : int, default=1
            Number of parallel jobs to run.
        parallelization_method : str, default='multithreading'
            Method of parallelization ('multithreading' or 'multiprocessing').
        memmap_folder: str, default='memmap'
            Folder used for storing temporary memmap files
        """
        self.K_range = K_range
        self.n_iterations = n_iterations
        self.subsampling = subsampling
        self.clusterer = clusterer
        self.clusterer_options = clusterer_options

        self.consensus_matrix_analysis = consensus_matrix_analysis
        self.PAC_interval = PAC_interval
        self.plot_cdf = plot_cdf
        self.agg_clustering_linkage = agg_clustering_linkage

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallelization_method = parallelization_method
        self.memmap_folder = Path(memmap_folder)
        if self.memmap_folder.exists():
            for _temp in self.memmap_folder.iterdir():
                _temp.unlink()

        if self.clusterer is None:
            print('KMeans is set as default clusterer')
            self.clusterer = KMeans()

    def fit(self, X):
        """
        Fit the ConsensusClustering model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        self._N, _ = X.shape
        self._dtype = np.uint8 if self.n_iterations < 256 else np.uint16
        self.cdf_at_K_data = dict()

        resampling_indices, Iij = self._subsample_X(X)

        for K in self.K_range:
            self._K = K
            self._set_clusterer_K()
            indices_iterator = tqdm(resampling_indices,
                                    desc=f'Consensus clustering with {self._K} clusters')

            if self.n_jobs == 1:
                Mij = self._run_n_jobs(X, indices_iterator, run_type='single')
            else:
                if self.parallelization_method == 'multithreading':
                    Mij = self._run_n_jobs(X, indices_iterator, run_type='threads')

                elif self.parallelization_method == 'multiprocessing':
                    Mij = self._run_n_jobs(X, indices_iterator, run_type='processes')

            self.cdf_at_K_data[K] = self._analyse_Cij(Mij, Iij)

            gc.collect()
            if self.parallelization_method == 'multiprocessing':
                Mij._mmap.close()

        if self.plot_cdf:
            self._plot_cdf()

        return self

    def _create_Mij_memmap(self):
        """
        Create writable shared memory Mij matrix.

        Returns:
        --------
        memmap : numpy.memmap
            Writable shared memory Mij matrix.
        """
        memmap_file = os.path.join(self.memmap_folder, f'_temp_{self._K}')

        try:
            os.mkdir(self.memmap_folder)
        except FileExistsError:
            pass

        memmap = np.memmap(
            memmap_file,
            dtype=self._dtype,
            shape=(self._N, self._N),
            mode='w+',
        )
        return memmap

    def _run_n_jobs(self, X, indices_iterator, run_type):
        """
        Clusterize subsample using clusterer.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        indices_iterator : iterator
            Iterator yielding subsampling indices.
        run_type : str
            Type of parallelization ('single', 'threads', or 'processes').

        Returns:
        --------
        Mij : array-like of shape (n_samples, n_samples)
            Consensus matrix.
        """
        if run_type == 'single':
            Mij = np.zeros((self._N, self._N), dtype=self._dtype)
            for indices in indices_iterator:
                self._fit_subsample(X, indices, Mij)

        elif run_type == 'threads':
            Mij = np.zeros((self._N, self._N), dtype=self._dtype)
            Parallel(n_jobs=self.n_jobs, prefer='threads')\
                (delayed(self._fit_subsample)(X, indices, Mij)\
                     for indices in indices_iterator)

        elif run_type == 'processes':
            Mij = self._create_Mij_memmap()
            Parallel(n_jobs=self.n_jobs, prefer='processes')\
                (delayed(self._fit_subsample)(X, indices, Mij)\
                     for indices in indices_iterator)
        else:
            raise RuntimeError(f'unknown parallelization method: {self.parallelization_method}')

        return Mij

    def _set_clusterer_K(self):
        """
        Set clusterer options.
        """
        if hasattr(self.clusterer, 'n_clusters'):
            self.clusterer.n_clusters = self._K
        elif hasattr(self.clusterer, 'n_components'):
            self.clusterer.n_components = self._K
        else:
            raise AttributeError('clusterer has neither n_clusters nor n_components attribute')

        self.clusterer.set_params(
            random_state=self.random_state, **self.clusterer_options
        )

    def _get_subsampling_indices(self, X):
        """
        Get subsampling indices for X for a given number of iterations.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        resampling_indices : array-like
            Subsampling indices for X.
        """
        resampling_indices = []
        for i in range(self.n_iterations):
            random_state = check_random_state(self.random_state + i)
            resampling_indices.append(
                random_state.choice(
                    self._N,
                    size=int(self.subsampling * self._N),
                    replace=False
                )
            )

        return np.r_[resampling_indices]

    def _subsample_X(self, X):
        """
        Return subsampling indices for X and resampling matrix Iij.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        resampling_indices : array-like
            Subsampling indices for X.
        Iij : array-like of shape (n_samples, n_samples)
            Resampling matrix.
        """
        resampling_indices = self._get_subsampling_indices(X)
        resampling_matrix = np.zeros((self.n_iterations, self._N), dtype=self._dtype)
        iter_indices = np.arange(self.n_iterations).reshape(-1, 1)
        resampling_matrix[iter_indices, resampling_indices] = 1

        Iij = np.dot(resampling_matrix.T, resampling_matrix)
        del resampling_matrix

        return resampling_indices, Iij

    def _fit_subsample(self, X, indices, Mij):
        """
        Fit clusterer to subset of X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        indices : array-like
            Subsampling indices.
        Mij : array-like of shape (n_samples, n_samples)
            Consensus matrix.
        """
        clustering = self.clusterer.fit_predict(X[indices])

        clustering_matrix = np.zeros((self._K, self._N), dtype=self._dtype)
        clustering_matrix[clustering, indices] = 1

        mij = np.dot(clustering_matrix.T, clustering_matrix)
        del clustering_matrix

        Mij += mij

    def _get_consensus_labels(self, Cij):
        """
        Fit to get consensus labels.

        Parameters:
        -----------
        Cij : array-like of shape (n_samples, n_samples)
            Consensus matrix.

        Returns:
        --------
        labels : array-like of shape (n_samples,)
            Consensus labels.
        """
        agg_clusterer = AgglomerativeClustering(
            n_clusters=self._K,
            linkage=self.agg_clustering_linkage,
            affinity='manhattan'
        )

        labels = agg_clusterer.fit_predict(Cij)

        return labels

    def _get_cdf_data(self, Cij, bins=20):
        """
        Get characteristics of consensus matrix values cdf.

        Parameters:
        -----------
        Cij : array-like of shape (n_samples, n_samples)
            Consensus matrix.
        bins : int, default=20
            Number of bins for histogram.

        Returns:
        --------
        hist : array-like
            Histogram of consensus values.
        cdf : array-like
            Cumulative distribution function.
        bin_edges : array-like
            Bin edges.
        pac_area : float
            PAC area.
        """
        consensus_values = np.triu(Cij, k=1).ravel()
        hist, bin_edges = np.histogram(
            consensus_values,
            bins=bins,
            range=(0, 1),
            density=True
        )

        dbin = bin_edges[1] - bin_edges[0]
        cdf = np.cumsum(hist) * dbin

        u1, u2 = self.PAC_interval
        u1_ind = int(u1 / dbin)
        u2_ind = int(u2 / dbin)
        pac_area = cdf[u2_ind - 1] - cdf[u1_ind]

        return hist, cdf, bin_edges, pac_area

    def _analyse_Cij(self, Mij, Iij):
        """
        Analyse consensus matrix.

        Parameters:
        -----------
        Mij : array-like of shape (n_samples, n_samples)
            Consensus matrix.
        Iij : array-like of shape (n_samples, n_samples)
            Resampling matrix.

        Returns:
        --------
        dict
            Dictionary containing analysis results.
        """
        Cij = np.divide(Mij, Iij + 1e-6, dtype=np.float32)
        np.fill_diagonal(Cij, 1.0)

        consensus_labels = []  # self._get_consensus_labels(Cij, K)
        hist, cdf, bin_edges, pac_area = self._get_cdf_data(Cij)

        return dict(
            consensus_labels=consensus_labels,
            hist=hist,
            cdf=cdf,
            bin_edges=bin_edges,
            pac_area=pac_area,
            mij=Mij,
            iij=Iij,
            cij=Cij,
        )

    def _plot_cdf(self):
        """
        Plot cdfs.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4), dpi=120)

        for K, data in self.cdf_at_K_data.items():
            x = data['bin_edges']
            y = [0] + [v for v in data['cdf']]
            plt.plot(x, y, marker='o', markersize=2.5,
                     label=f'K: {K}', linewidth=2.0)

        plt.vlines(self.PAC_interval,
                   *plt.ylim(),
                   colors='k', linestyles='dashed', lw=1.5)
        plt.xlabel('consensus index value')
        plt.ylabel('CDF')
        plt.legend()
        plt.tight_layout()
        plt.show()
