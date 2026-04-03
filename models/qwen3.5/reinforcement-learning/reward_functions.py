"""
Reward functions per il training GRPO su generazione VGDL.

Valutano le completions generate dal modello su diversi criteri:
  - Eseguibilità VGDL (parser py-vgdl)
  - Struttura corretta (4 sezioni obbligatorie + BasicGame)
  - Classi sprite valide (ontologia py-vgdl)
  - Effetti di interazione validi (ontologia py-vgdl)
  - Condizioni di terminazione valide (ontologia py-vgdl)
  - Uso di EOS al posto di keyword di bordo non valide
"""

import re

# ========================
# Ontologia VGDL (estratta da py-vgdl/vgdl/ontology.py)
# ========================
# Pre-compiled regex patterns (compilati una sola volta al modulo load, non ad ogni chiamata)
_RE_SECTION_CACHE = {}

def _section_re(section_name):
    if section_name not in _RE_SECTION_CACHE:
        _RE_SECTION_CACHE[section_name] = re.compile(rf'^\s*{section_name}\b', re.MULTILINE)
    return _RE_SECTION_CACHE[section_name]

_RE_SPRITE_CLASS    = re.compile(r'>\s*([A-Z][A-Za-z]+)')
_RE_INTERACTION_SEC = re.compile(r'InteractionSet(.*?)(?=TerminationSet|LevelMapping|SpriteSet|\Z)', re.DOTALL)
_RE_EFFECT          = re.compile(r'>\s*([a-z][A-Za-z]+)')
_RE_TERM_SEC        = re.compile(r'TerminationSet(.*?)(?=InteractionSet|LevelMapping|SpriteSet|\Z)', re.DOTALL)
_RE_TERM_COND       = re.compile(r'^\s+([A-Z][A-Za-z]+)', re.MULTILINE)
_RE_EOS             = re.compile(r'\bEOS\b')
_RE_BASIC_GAME_LINE = re.compile(r'BasicGame[^\n]*')
_RE_PARAM_KEY       = re.compile(r'(\w+)\s*=')
_RE_INVALID_BOUND   = {kw: re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in {"edge", "screen", "wall_bound", "boundary"}}

VALID_SPRITE_CLASSES = {
    # Sprite statici / immovable
    "Immovable", "Passive", "ResourcePack", "Flicker", "Spreader",
    # Sprite orientati
    "Conveyor", "Missile", "OrientedFlicker", "Walker", "WalkJumper",
    # NPC
    "RandomNPC", "RandomInertial", "RandomMissile", "ErraticMissile",
    "Bomber", "Chaser", "Fleeing", "AStarChaser",
    # Producer
    "SpawnPoint", "Portal",
    # Avatar (player-controlled)
    "MovingAvatar", "HorizontalAvatar", "VerticalAvatar", "FlakAvatar",
    "OrientedAvatar", "RotatingAvatar", "RotatingFlippingAvatar",
    "NoisyRotatingFlippingAvatar", "ShootAvatar", "AimedAvatar",
    "AimedFlakAvatar", "InertialAvatar", "MarioAvatar",
}

VALID_INTERACTION_EFFECTS = {
    "killSprite", "cloneSprite", "transformTo",
    "stepBack", "undoAll",
    "bounceForward", "bounceDirection", "wallBounce", "wallStop",
    "conveySprite", "windGust", "slipForward", "pullWithIt",
    "attractGaze", "turnAround", "reverseDirection", "flipDirection",
    "wrapAround", "teleportToExit",
    "killIfSlow", "killIfFromAbove", "killIfAlive",
    "killIfHasMore", "killIfHasLess",
    "killIfOtherHasMore", "killIfOtherHasLess",
    "collectResource", "changeResource", "spawnIfHasMore",
}

VALID_TERMINATION_CONDITIONS = {
    "Timeout", "SpriteCounter", "MultiSpriteCounter",
}

MANDATORY_SECTIONS = ["SpriteSet", "LevelMapping", "InteractionSet", "TerminationSet"]

# Keyword NON valide per il bordo dello schermo (si deve usare EOS)
INVALID_BOUNDARY_KEYWORDS = {"edge", "screen", "wall_bound", "boundary"}

# Parametri validi per la riga BasicGame
VALID_BASIC_GAME_PARAMS = {"block_size", "fps", "num_sprites"}


# ========================
# Validazione VGDL su stringa
# (adattamento di evaluation/check_vgdl_executability.py)
# ========================

# Singleton VGDLParser: evita di reinstanziare il parser (e ricaricare l'ontologia)
# ad ogni completion. Il parser è stateless rispetto alle chiamate a parseGame.
_VGDL_PARSER = None

def _get_parser():
    global _VGDL_PARSER
    if _VGDL_PARSER is None:
        from vgdl.core import VGDLParser  # type: ignore
        _VGDL_PARSER = VGDLParser()
    return _VGDL_PARSER


def _parse_vgdl_string(vgdl_str: str):
    """Parsa una stringa VGDL. Restituisce (game | None, errori: list)."""
    try:
        game = _get_parser().parseGame(vgdl_str.strip())
        return game, []
    except Exception as e:
        return None, [f"Parsing error: {e}"]


def _validate_vgdl_string(vgdl_str: str):
    """Valida una stringa VGDL. Restituisce (valid: bool, errori: list)."""
    game, errors = _parse_vgdl_string(vgdl_str)
    if game is None:
        return False, errors

    try:
        sprites = set(game.sprite_constr.keys())
    except Exception:
        sprites = set()

    # Controlla sprite nelle interazioni
    try:
        valid_sprites = set(game.sprite_constr.keys())
        special_sprites = {"EOS", "wall"}
        for _, (_, _, stypes) in game.sprite_constr.items():
            valid_sprites.update(stypes)
        for interaction in game.collision_eff:
            s1, s2 = interaction[0], interaction[1]
            if s1 not in valid_sprites and s1 not in special_sprites:
                errors.append(f"Interaction uses undefined sprite: {s1}")
            if s2 not in valid_sprites and s2 not in special_sprites:
                errors.append(f"Interaction uses undefined sprite: {s2}")
    except Exception:
        pass

    # Controlla sprite nelle terminazioni
    try:
        for term in game.terminations:
            if hasattr(term, "stype") and term.stype and term.stype not in sprites:
                errors.append(f"Termination references undefined sprite: {term.stype}")
    except Exception:
        pass

    return len(errors) == 0, errors


# ========================
# Reward Functions
# ========================

def reward_executability(completions, **kwargs):
    """
    Reward principale: 1.0 se il codice VGDL è eseguibile (parser + check semantici),
    0.0 altrimenti.
    """
    rewards = []
    for text in completions:
        valid, _ = _validate_vgdl_string(text)
        rewards.append(1.0 if valid else 0.0)
    return rewards


def reward_structure(completions, **kwargs):
    """
    Reward per la struttura corretta del VGDL (0.0 – 1.0):
      +0.2  se inizia con 'BasicGame'
      +0.2  per ogni sezione obbligatoria presente (SpriteSet, LevelMapping,
             InteractionSet, TerminationSet)
    """
    rewards = []
    for text in completions:
        score = 0.0
        if text.strip().startswith("BasicGame"):
            score += 0.2
        for section in MANDATORY_SECTIONS:
            if _section_re(section).search(text):
                score += 0.2
        rewards.append(score)
    return rewards


def reward_valid_sprite_classes(completions, **kwargs):
    """
    Reward per l'uso di classi sprite valide dall'ontologia py-vgdl.
    Score = frazione di '> ClassName' che sono classi valide.
    0.5 se nessuna classe trovata (score neutro).
    """
    rewards = []
    for text in completions:
        # SpriteSet lines: "name > ClassName [params...]"
        matches = _RE_SPRITE_CLASS.findall(text)
        if not matches:
            rewards.append(0.5)
            continue
        valid_count = sum(1 for cls in matches if cls in VALID_SPRITE_CLASSES)
        rewards.append(valid_count / len(matches))
    return rewards


def reward_valid_interactions(completions, **kwargs):
    """
    Reward per l'uso di effetti di interazione validi dall'ontologia py-vgdl.
    Score = frazione di 'sprite1 sprite2 > effectName' che usano effetti validi.
    0.5 se nessun effetto trovato nella sezione InteractionSet.
    """
    rewards = []
    for text in completions:
        inter_match = _RE_INTERACTION_SEC.search(text)
        if not inter_match:
            rewards.append(0.5)
            continue
        section_text = inter_match.group(1)
        effects = _RE_EFFECT.findall(section_text)
        if not effects:
            rewards.append(0.5)
            continue
        valid_count = sum(1 for e in effects if e in VALID_INTERACTION_EFFECTS)
        rewards.append(valid_count / len(effects))
    return rewards


def reward_valid_terminations(completions, **kwargs):
    """
    Reward per l'uso di condizioni di terminazione valide (Timeout, SpriteCounter,
    MultiSpriteCounter).
    Score = frazione di condizioni usate che sono valide.
    0.5 se nessuna condizione trovata (score neutro).
    """
    rewards = []
    for text in completions:
        term_match = _RE_TERM_SEC.search(text)
        if not term_match:
            rewards.append(0.5)
            continue
        section_text = term_match.group(1)
        conditions = _RE_TERM_COND.findall(section_text)
        if not conditions:
            rewards.append(0.5)
            continue
        valid_count = sum(1 for c in conditions if c in VALID_TERMINATION_CONDITIONS)
        rewards.append(valid_count / len(conditions))
    return rewards


def reward_eos_boundary(completions, **kwargs):
    """
    Reward per l'uso corretto di EOS per i bordi dello schermo:
      1.0  se usa EOS (corretto)
      0.0  se usa keyword non valide (edge, screen, wall_bound, boundary)
      0.5  se non menziona bordi (neutro)
    """
    rewards = []
    for text in completions:
        has_eos = bool(_RE_EOS.search(text))
        has_invalid = any(rx.search(text) for rx in _RE_INVALID_BOUND.values())
        if has_invalid:
            rewards.append(0.0)
        elif has_eos:
            rewards.append(1.0)
        else:
            rewards.append(0.5)
    return rewards


def reward_no_undefined_params(completions, **kwargs):
    """
    Penalizza parametri non riconosciuti da py-vgdl nella riga BasicGame.
    Score = -0.1 per ogni parametro sconosciuto trovato.
    0.0 se la riga BasicGame non è presente o non ha parametri unknown.
    """
    rewards = []
    for text in completions:
        basic_game_match = _RE_BASIC_GAME_LINE.search(text)
        if not basic_game_match:
            rewards.append(0.0)
            continue
        found = _RE_PARAM_KEY.findall(basic_game_match.group())
        unknown = [p for p in found if p not in VALID_BASIC_GAME_PARAMS]
        rewards.append(-0.1 * len(unknown))
    return rewards


# Lista ordinata di reward functions (GRPOTrainer le somma automaticamente)
REWARD_FUNCTIONS = [
    reward_executability,          # peso implicito: 1x (il più importante)
    reward_structure,              # peso implicito: 1x
    reward_valid_sprite_classes,   # peso implicito: 1x
    reward_valid_interactions,     # peso implicito: 1x
    reward_valid_terminations,     # peso implicito: 1x
    reward_eos_boundary,           # peso implicito: 1x
    reward_no_undefined_params,    # penalità per parametri BasicGame non validi
]
