import logging


class CompositeRouter:
    """Routes between multiple ProviderRouter instances.

    Strategy: choose the first healthy provider and delegate the request.
    This is intentionally simple for the MVP and easy to extend later.
    """

    def __init__(self, providers: list):
        self.providers = providers or []

    async def handle(self, op: str, payload: dict, **kwargs):
        last_exc = None
        for provider in self.providers:
            try:
                if await provider.is_healthy():
                    return await provider.handle(op, payload, **kwargs)
            except (AttributeError, RuntimeError, OSError, TypeError) as e:
                logging.debug("provider health check/handle failed: %s", e)
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("No healthy providers available")
