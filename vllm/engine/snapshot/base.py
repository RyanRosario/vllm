# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import shutil
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSnapshotProvider(ABC):
    @abstractmethod
    def trigger(self) -> None:
        """Cloud-specific logic to snapshot the Pod/GPU."""
        pass


def is_on_mount_or_link(path: str) -> bool:
    path = os.path.expanduser(path)
    if os.path.islink(path):
        return True
    path = os.path.abspath(path)
    while path and path != os.path.sep:
        if os.path.ismount(path):
            return True
        path = os.path.dirname(path)
    return False


class GKESnapshotProvider(BaseSnapshotProvider):
    def trigger(self) -> None:
        # Clear out duplicate weights if not stored on a mount or symlink
        try:
            hf_cache = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(hf_cache):
                if not is_on_mount_or_link(hf_cache):
                    shutil.rmtree(hf_cache, ignore_errors=True)
                else:
                    logger.info(
                        "Hugging Face cache is on a mount point or "
                        "symlink; skipping deletion to avoid "
                        "permanently evicting weights."
                    )
        except Exception as e:
            logger.error("Could not delete locally stored weights: %s", e)

        # Write 1 to /proc/sys/checkpoint.
        # Using the same descriptor try to read from the file
        try:
            fd = os.open("/proc/gvisor/checkpoint", os.O_RDWR)
            try:
                os.write(fd, b"1")
                os.read(fd, 1)
            finally:
                os.close(fd)
        except Exception as e:
            logger.error("gVisor checkpoint I/O error occurred: %s", e)
            raise RuntimeError("gVisor checkpoint triggering failed") from e
