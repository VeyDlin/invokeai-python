# Path: invoke\invoke.py
import asyncio
import aiohttp

from .api.utilities.utilities_api import UtilitiesApi
from .api.models.models_api import ModelsApi, BaseModels, ModelType
from .api.images.images_api import ImagesApi, ImageOrigin, Categories
from .api.boards.boards_api import BoardsApi
from .api.app.app_api import AppApi
from .api.queue.queue_api import QueueApi
from .api.queue.schema import EnqueueBatch
from .api.download_queue.download_queue_api import DownloadQueueApi


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


    async def wait_batch(self, batch: EnqueueBatch, delay: float = 0.1):
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