import os
import warnings
from tqdm import tqdm
import logging
try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import

logging.basicConfig(level=logging.INFO)


model_urls = {
    "small": {
        "text": "https://drive.google.com/u/0/uc?id=1wJRqkmc371zBh4_OtmrsxabnqkqahK0o&export=download",
        "weights": "https://drive.google.com/u/0/uc?id=1UFJDj6EJMgcnjC1v_s8b8pIzeefJ4nLv&export=download",
        "size": 32116
    },
    "normal": {
        "text": "https://drive.google.com/u/0/uc?id=1276R4INKlaEhprSt8SLOkmzSaQcHK92M&export=download",
        "weights": "https://drive.google.com/u/0/uc?id=1d6StXVANkORfoQv19fOyH1fLl07x-Ge-&export=download",
        "size": 82903
    },
    "large": {
        "text": "https://drive.google.com/u/0/uc?id=1z8a3R_AsP22eVu0pbyA_K2n1G_s5Icso&export=download",
        "weights": "https://www.dropbox.com/s/au0f2gf2twq0pyq/GENet_large.pth?dl=1",
        "size": 121866,
    },
}


def download(
        url: str,
        path: str = None,
        size: int = None,
        overwrite: bool = False,
        retries: int = 5,
        verify_ssl: bool = True
) -> str:
    """
    Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    size : str, optional
        Size of the weights file
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns
    -------
    str
        The file path of the downloaded file.
    """

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. " \
            "Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                logging.info(f"\tDownloading {fname} from {url}...")
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024), total=size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    logging.info(f"\tdownload failed, retrying, {retries} attempt{'s' if retries > 1 else ''} left")

    return fname
