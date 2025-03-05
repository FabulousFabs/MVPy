import joblib
import contextlib
from tqdm.auto import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class Progressbar:
    """Simple class for progress bars that can be enabled or disabled.
    
    Parameters
    ----------
    enabled : bool, default=True
        Whether to enable the progress bar.
    **kwargs : Any
        Additional arguments for tqdm.
    """
    
    def __new__(self, enabled: bool = True, **kwargs):
        """Instantiate the progressbar, either as tqdm or dummy.
        
        Parameters
        ----------
        enabled : bool, default=True
            Whether to enable the progress bar.
        kwargs : Any
            Additional arguments for tqdm.
        
        Returns
        -------
        Union[tqdm, Progressbar]
            The progressbar context.
        """
        
        if enabled:
            return tqdm_joblib(tqdm(**kwargs))
        
        return super().__new__(self)
    
    def __enter__(self):
        """Vacant."""
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Vacant."""
        
        pass