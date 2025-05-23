from typing import Type
from .delegating_citizen import DelegatingFC
from .citizen import CitizenFC, Citizen, RouterFC
from .vision_citizen import VisionCitizen, DelegatingVisionCitizen, VisionRouter


citizens: tuple[Type[Citizen], ...] = (DelegatingFC, CitizenFC, RouterFC, VisionCitizen, DelegatingVisionCitizen, VisionRouter)

name_to_citizen: dict[str, Type[Citizen]] = {c.__name__: c for c in citizens}