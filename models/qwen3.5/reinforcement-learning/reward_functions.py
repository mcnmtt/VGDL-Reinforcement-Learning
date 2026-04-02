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


# ========================
# Validazione VGDL su stringa
# (adattamento di evaluation/check_vgdl_executability.py)
# ========================

def _parse_vgdl_string(vgdl_str: str):
    """Parsa una stringa VGDL. Restituisce (game | None, errori: list)."""
    try:
        from vgdl.core import VGDLParser  # type: ignore
        parser = VGDLParser()
        game = parser.parseGame(vgdl_str.strip())
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
            if re.search(rf'^\s*{section}\b', text, re.MULTILINE):
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
        matches = re.findall(r'>\s*([A-Z][A-Za-z]+)', text)
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
        inter_match = re.search(
            r'InteractionSet(.*?)(?=TerminationSet|LevelMapping|SpriteSet|\Z)',
            text, re.DOTALL
        )
        if not inter_match:
            rewards.append(0.5)
            continue
        section_text = inter_match.group(1)
        # effetti iniziano con lettera minuscola dopo ">"
        effects = re.findall(r'>\s*([a-z][A-Za-z]+)', section_text)
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
        term_match = re.search(
            r'TerminationSet(.*?)(?=InteractionSet|LevelMapping|SpriteSet|\Z)',
            text, re.DOTALL
        )
        if not term_match:
            rewards.append(0.5)
            continue
        section_text = term_match.group(1)
        # condizioni iniziano con lettera maiuscola (indentate)
        conditions = re.findall(r'^\s+([A-Z][A-Za-z]+)', section_text, re.MULTILINE)
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
        has_eos = bool(re.search(r'\bEOS\b', text))
        has_invalid = any(
            bool(re.search(rf'\b{kw}\b', text, re.IGNORECASE))
            for kw in INVALID_BOUNDARY_KEYWORDS
        )
        if has_invalid:
            rewards.append(0.0)
        elif has_eos:
            rewards.append(1.0)
        else:
            rewards.append(0.5)
    return rewards


# Lista ordinata di reward functions (GRPOTrainer le somma automaticamente)
REWARD_FUNCTIONS = [
    reward_executability,        # peso implicito: 1x (il più importante)
    reward_structure,            # peso implicito: 1x
    reward_valid_sprite_classes, # peso implicito: 1x
    reward_valid_interactions,   # peso implicito: 1x
    reward_valid_terminations,   # peso implicito: 1x
    reward_eos_boundary,         # peso implicito: 1x
]
