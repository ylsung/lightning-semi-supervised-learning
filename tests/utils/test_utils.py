import os
import pytest
import subprocess
from lightning_ssl.utils.utils import makedirs


def test_makedirs(tmpdir):
    folder = os.path.join(tmpdir, "a/b")
    makedirs(folder)
    assert os.path.exists(folder)
