from typing import Type
from .delegating_citizen import DelegatingFC
from .citizen import CitizenFC, Citizen


citizens: tuple[Type[Citizen], ...] = (DelegatingFC, CitizenFC)

name_to_citizen: dict[str, Type[Citizen]] = {c.__name__: c for c in citizens}