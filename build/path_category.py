CATS = {
    'block_item': ('重点', '已有方块与物品机制'),
    'entity':     ('重点', '已有实体与玩家机制'),
    'worldgen':   ('重点', '已有地形生成的修改'),
    'world_run':  ('重点', '世界运行（区块加载、刻调度）'),
    'network':    ('次重点', '网络协议机制变化'),
    'admin':      ('次重点', '管理配置能力'),
    'ui':         ('不重要', '界面与显示调整'),
    'command':    ('不重要', '命令与数据包常规调整'),
    'protocol':   ('不重要', '协议细节调整'),
}
TIER_ORDER = {'重点': 0, '次重点': 1, '不重要': 2}

RULES = [
    ('net/minecraft/client/', 'ui'), ('com/mojang/blaze3d/', 'ui'), ('com/mojang/realmsclient/', 'ui'),
    ('com/mojang/renderpearl/', 'ui'), ('assets/', 'ui'), ('net/minecraft/core/particles/', 'ui'),
    ('net/minecraft/sounds/', 'ui'),
    ('net/minecraft/server/commands/', 'command'), ('net/minecraft/commands/', 'command'),
    ('net/minecraft/advancements/', 'command'), ('net/minecraft/gametest/', 'command'),
    ('net/minecraft/data/advancements/', 'command'), ('data/minecraft/advancements/', 'command'),
    ('data/minecraft/advancement/', 'command'), ('net/minecraft/server/packs/', 'command'),
    ('net/minecraft/world/scores/', 'command'),
    ('net/minecraft/network/', 'network'), ('net/minecraft/server/network/', 'network'),
    ('version.json', 'protocol'),
    ('net/minecraft/server/dedicated/', 'admin'), ('net/minecraft/server/players/', 'admin'),
    ('net/minecraft/server/jsonrpc/', 'admin'), ('net/minecraft/server/rcon/', 'admin'),
    ('net/minecraft/world/level/GameRules.java', 'admin'), ('net/minecraft/world/level/gamerules/', 'admin'),
    ('net/minecraft/world/level/levelgen/', 'worldgen'), ('net/minecraft/world/level/biome/', 'worldgen'),
    ('net/minecraft/data/worldgen/', 'worldgen'), ('data/minecraft/worldgen/', 'worldgen'),
    ('data/minecraft/tags/worldgen/', 'worldgen'), ('data/minecraft/structures/', 'worldgen'),
    ('data/minecraft/structure/', 'worldgen'), ('data/minecraft/dimension_type/', 'worldgen'),
    ('net/minecraft/world/level/dimension/', 'worldgen'),
    ('net/minecraft/server/level/', 'world_run'), ('net/minecraft/world/level/chunk/', 'world_run'),
    ('net/minecraft/world/ticks/', 'world_run'), ('net/minecraft/world/level/entity/', 'world_run'),
    ('net/minecraft/world/level/border/', 'world_run'), ('net/minecraft/world/level/lighting/', 'world_run'),
    ('net/minecraft/world/level/chunk/storage/', None),
    ('net/minecraft/world/entity/', 'entity'), ('net/minecraft/world/effect/', 'entity'),
    ('net/minecraft/world/damagesource/', 'entity'), ('net/minecraft/world/food/', 'entity'),
    ('net/minecraft/world/attribute/', 'entity'), ('net/minecraft/world/level/pathfinder/', 'entity'),
    ('net/minecraft/server/level/ServerPlayer.java', 'entity'),
    ('net/minecraft/server/level/ServerPlayerGameMode.java', 'entity'),
    ('net/minecraft/world/level/NaturalSpawner.java', 'entity'), ('net/minecraft/world/level/BaseSpawner.java', 'entity'),
    ('net/minecraft/world/level/SpawnData.java', 'entity'),
    ('data/minecraft/tags/entity_type', 'entity'), ('data/minecraft/tags/damage_type/', 'entity'),
    ('data/minecraft/loot_tables/entities/', 'entity'), ('data/minecraft/loot_table/entities/', 'entity'),
    ('data/minecraft/villager_trade/', 'entity'), ('data/minecraft/tags/game_events/', 'block_item'),
    ('net/minecraft/world/level/', 'block_item'), ('net/minecraft/world/item/', 'block_item'),
    ('net/minecraft/world/inventory/', 'block_item'), ('net/minecraft/core/dispenser/', 'block_item'),
    ('net/minecraft/core/cauldron/', 'block_item'), ('net/minecraft/recipebook/', 'block_item'),
    ('net/minecraft/data/loot/', 'block_item'), ('net/minecraft/data/recipes/', 'block_item'),
    ('net/minecraft/data/tags/', 'block_item'),
    ('data/minecraft/tags/block', 'block_item'), ('data/minecraft/tags/item', 'block_item'),
    ('data/minecraft/loot_tables/', 'block_item'), ('data/minecraft/loot_table/', 'block_item'),
    ('data/minecraft/recipes/', 'block_item'), ('data/minecraft/recipe/', 'block_item'),
    ('data/minecraft/enchantment/', 'block_item'),
    ('net/minecraft/world/level/storage/', None), ('net/minecraft/world/level/storage/loot/', 'block_item'),
    ('net/minecraft/world/level/saveddata/', None),
]
RULES.sort(key=lambda r: -len(r[0]))


def path_cat(path):
    for prefix, cat in RULES:
        if path.startswith(prefix):
            return cat
    return None


def doc_cat(paths, method):
    cats = [c for c in (path_cat(p) for p in paths) if c]
    if not cats:
        return None
    order = list(CATS)
    if method == 'max':
        return min(cats, key=lambda c: (TIER_ORDER[CATS[c][0]], order.index(c)))
    n = {c: cats.count(c) for c in set(cats)}
    return min(n, key=lambda c: (-n[c], TIER_ORDER[CATS[c][0]], order.index(c)))
