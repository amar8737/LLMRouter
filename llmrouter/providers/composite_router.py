import logging
from ..exceptions import NoHealthyClientError


class CompositeRouter:
    """Routes between multiple ProviderRouter instances.

    Strategy: try providers in order and avoid TOCTOU by invoking `handle`
    directly and catching provider-level errors.
    """

    def __init__(self, providers: list, metrics=None):
        self.providers = providers or []
        self.metrics = metrics

    async def handle(self, op: str, payload: dict, **kwargs):
        last_exc = None
        for provider in self.providers:
            provider_name = getattr(provider, "name", "unknown")
            try:
                result = await provider.handle(op, payload, **kwargs)
                if self.metrics:
                    self.metrics.incr(f"provider.success.{provider_name}")
                return result
            except NoHealthyClientError as e:
                logging.debug("provider %s has no healthy clients: %s", provider_name, e)
                last_exc = e
                if self.metrics:
                    self.metrics.incr(f"provider.no_healthy.{provider_name}")
                continue
            except Exception as e:
                logging.exception("provider %s failed: %s", provider_name, e)
                last_exc = e
                if self.metrics:
                    self.metrics.incr(f"provider.error.{provider_name}")
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("No healthy providers available")
