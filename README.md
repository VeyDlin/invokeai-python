# invokeai-python

`invokeai-python` is a powerful library designed for working with graphs, models, and API integrations, aimed at simplifying the creation and management of tasks related to generative AI.

## Installation

1. Ensure you have Python version 3.6 or higher installed.
2. Install the library using pip:

```bash
pip install invokeai-python
```

## Usage

### Importing Key Components

```python
from invoke import Invoke
from invoke.api import BaseModels, ModelType
```

### Example 1: Querying Models

```python
import asyncio
from invoke import Invoke
from invoke.api import BaseModels, ModelType

async def main():
    invoke = Invoke()

    print("Waiting for invoke...")
    version = await invoke.wait_invoke()
    print(f"Version: {version}")

    models = await invoke.models.list(base_models=[BaseModels.SDXL], model_type=[ModelType.Main])
    print(models)

if __name__ == "__main__":
    asyncio.run(main())
```

## Contributing
If you would like to contribute, feel free to submit an issue or a pull request on the [GitHub repository](https://github.com/veydlin/invokeai-python).

## License
MIT License