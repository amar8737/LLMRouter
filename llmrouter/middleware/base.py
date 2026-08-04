class BaseMiddleware:
    async def before_request(self, op: str, payload: dict):
        return payload

    async def after_response(self, op: str, payload: dict, response: dict):
        return response
