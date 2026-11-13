"""Every scene, in order. A scene is a mixin class with a `v_<name>(cv, P, h, w)` method; App inherits them all."""
from .shared import SharedScene
from .spectrum import SpectrumScene
from .city import CityScene
from .forest import ForestScene
from .cafe import CafeScene
from .ocean import OceanScene
from .blossom import BlossomScene
from .lava import LavaScene
from .cat import CatScene
from .koi import KoiScene
from .train import TrainScene
from .aquarium import AquariumScene
from .rooftop import RooftopScene
from .planet import PlanetScene

# (name, method, one-line description) — the position in this list is the scene's number
SCENES = [
    ('spectrum', 'v_spectrum', 'pastel bars and peak caps'),
    ('rainy city', 'v_city', 'skyline, rain and puddles'),
    ('forest', 'v_forest', 'pines, campfire and an owl'),
    ('coffee shop', 'v_cafe', 'regulars, lamps and a cat'),
    ('ocean', 'v_ocean', 'sailboat and lighthouse'),
    ('cherry blossom', 'v_blossom', 'petals falling below Mount Fuji'),
    ('lava lamp', 'v_lava', 'wax that the bass heats up'),
    ('sleepy cat', 'v_cat', 'napping by the window'),
    ('koi pond', 'v_koi', 'koi seen from above'),
    ('night train', 'v_train', 'the countryside rushing by'),
    ('aquarium', 'v_aquarium', 'fish, plants and treasure'),
    ('rooftop', 'v_rooftop', 'string lights and laundry'),
    ('planet', 'v_planet', 'a world that turns with the tempo'),
]
VIZ_NAMES = [s[0] for s in SCENES]
SCENE_DESC = [s[2] for s in SCENES]
MIXINS = (SharedScene, SpectrumScene, CityScene, ForestScene, CafeScene, OceanScene, BlossomScene, LavaScene, CatScene, KoiScene, TrainScene, AquariumScene, RooftopScene, PlanetScene,)
