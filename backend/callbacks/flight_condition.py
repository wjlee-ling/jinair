from langchain_core.callbacks import BaseCallbackHandler


class FlightConditionCallbackHandler(BaseCallbackHandler):
    """Callback for the flight condition entity extraction chain."""

    def __init__(self):
        self.entities = {}

    def update(self, entities: dict):
        self.entities.update(entities)

    def on_chain_end(self, outputs, **kwargs):
        d = outputs.dict()
        self.entities.update(
            {
                "origin": d.get("origin", ""),
                "destination": d.get("destination", ""),
                "date": d.get("date", ""),
                "persons": d.get("persons", 1),
            }
        )
