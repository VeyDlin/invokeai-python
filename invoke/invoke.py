# Path: invoke\invoke.py
import asyncio
import aiohttp

from .api.utilities import UtilitiesApi
from .api.models import ModelsApi
from .api.images import ImagesApi
from .api.boards import BoardsApi
from .api.app import AppApi
from .api.queue import QueueApi, EnqueueBatch
from .api.download_queue import DownloadJobStatus, DownloadQueueApi


class Invoke:
    _client: aiohttp.ClientSession

    host: str
    utilities: UtilitiesApi
    models: ModelsApi
    images: ImagesApi
    boards: BoardsApi
    app: AppApi
    queue: QueueApi
    downloadQueue: DownloadQueueApi


    def __init__(self, host: str = "http://127.0.0.1:9090",):
        self._client = aiohttp.ClientSession()
        self.host = host
        self.utilities = UtilitiesApi(self._client, host)
        self.models = ModelsApi(self._client, host)
        self.images = ImagesApi(self._client, host)
        self.boards = BoardsApi(self._client, host)
        self.app = AppApi(self._client, host)
        self.queue = QueueApi(self._client, host)
        self.downloadQueue = DownloadQueueApi(self._client, host)


    def __del__(self):
        if self._client:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._client.close())
            else:
                loop.run_until_complete(self._client.close())
            self._client = None


    async def wait_invoke(self, delay: float = 0.1) -> str:
        while True:
            try:
                version = await self.app.version()
                return version.version
            except Exception:
                await asyncio.sleep(delay)


    async def wait_batch(self, batch: EnqueueBatch, delay: float = 0.1) -> None:
        batch_id = batch.batch.batch_id
        while True:
            await asyncio.sleep(delay)

            status = await self.queue.get_batch_status(batch_id)

            if status.failed == 1:
                # queue_list = await self.invoke.queue.list(limit=100)
                # error = queue_list.items[-1].error
                raise Exception("Invoke batch error")

            if status.canceled == 1:
                raise Exception("Invoke batch canceled")

            if status.completed == 1:
                break


    async def wait_install_models(self, delay: float = 0.1, raise_on_error: bool = False) -> None:
        while True:
            queue = await self.downloadQueue.list()
            statuses = [job.status for job in queue]

            if all(status == DownloadJobStatus.completed for status in statuses):
                return

            if raise_on_error:
                errors = [f"Error: {job.error}, Path: {job.download_path}" for job in queue if job.status in {DownloadJobStatus.cancelled, DownloadJobStatus.error}]
                if errors:
                    raise RuntimeError("One or more jobs failed or were cancelled:\n" + "\n".join(errors))

            await asyncio.sleep(delay)