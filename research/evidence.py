"""Read the versioned pilot graph and resolve quantities against pinned project exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PREFIX = Path("BM@N/FHCal/FHCal_GMM")

GROUPS = ("hypotheses", "results", "limitations", "claims", "materials")
STATUS = {"preliminary", "verified", "superseded"}


def content(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def sequence(items, label):
    require(isinstance(items, list) and items, f"empty or invalid {label}")
    return items


class Graph:
    def __init__(self, registry, literature):
        require(registry["version"] == 1, "unsupported graph version")
        self.groups = registry
        self.nodes = {}
        self.uses = defaultdict(set)
        self.files = defaultdict(set)
        self.publications = {}
        for group in GROUPS:
            for key, node in registry[group].items():
                self.add(key, node)
        for paper in literature["entries"]:
            key = paper["id"]
            self.add(key, paper)
            self.publications[key] = paper
            self.files[paper["bib_path"]].add(key)
            for link in paper["claims"]:
                self.edge(key, link["id"], registry["claims"])
        for key, node in registry["results"].items():
            self.edge(node["hypothesis"], key, registry["hypotheses"], incoming=True)
            require(node["status"] in STATUS, f"{key}: invalid status")
            for field in ("scope", "code_revision", "run", "configuration", "review"):
                require(node.get(field), f"{key}: missing {field}")
            for source in sequence(node["sources"], key):
                self.files[source["path"]].add(key)
        for group, field, target in [
            ("limitations", "evidence", "results"),
            ("claims", "evidence", "results"),
            ("claims", "limitations", "limitations"),
            ("materials", "claims", "claims"),
        ]:
            for key, node in registry[group].items():
                for parent in sequence(node[field], key):
                    self.edge(parent, key, registry[target], incoming=True)
        for key, node in registry["claims"].items():
            for field in ("scope", "title", "counterargument"):
                require(node.get(field), f"{key}: missing {field}")
        for key, node in registry["materials"].items():
            require(node.get("anchor") and node.get("path"), f"{key}: missing location")

    def add(self, key, node):
        require(key and key not in self.nodes, f"duplicate ID {key}")
        self.nodes[key] = node

    def edge(self, parent, child, allowed, *, incoming=False):
        reference = parent if incoming else child
        require(reference in allowed, f"invalid reference {reference} ({parent} -> {child})")
        self.uses[parent].add(child)

    def path(self, name):
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"invalid path: {name}")
        return REPO_ROOT / path.relative_to(PROJECT_PREFIX)

    def impact(self, target):
        start = {target} if target in self.nodes else self.files.get(target)
        require(start, f"unknown node or evidence path: {target}")
        seen = set(start)
        queue = deque(start)
        while queue:
            for child in self.uses[queue.popleft()] - seen:
                seen.add(child)
                queue.append(child)
        return sorted(seen)

    def values(self, result):
        node = self.groups["results"][result]
        require(node["status"] == "verified", f"{result}: {node['status']}")
        values = {}
        for source in node["sources"]:
            path = self.path(source["path"])
            data = path.read_bytes()
            require(
                hashlib.sha256(data).hexdigest() == source["sha256"],
                f"changed {source['path']}; review: {', '.join(self.impact(result))}",
            )
            if source["fields"]:
                document = json.loads(data)
                for key, address in source["fields"].items():
                    require(key not in values, f"{result}: duplicate metric {key}")
                    value = document
                    for part in address:
                        value = value[part]
                    values[key] = value
        return values

    def claim(self, key):
        claim = self.groups["claims"][key]
        limits = self.groups["limitations"]
        evidence = list(
            dict.fromkeys(
                claim["evidence"]
                + [ref for limit in claim["limitations"] for ref in limits[limit]["evidence"]]
            )
        )
        return {
            "claim": claim,
            "limitations": [limits[limit]["text"] for limit in claim["limitations"]],
            "results": {ref: self.values(ref) for ref in evidence},
            "sources": {ref: self.groups["results"][ref]["sources"] for ref in evidence},
            "literature": [
                {"id": key_lit, "bibkey": paper["bibkey"], "role": link["role"]}
                for key_lit, paper in self.publications.items()
                for link in paper["claims"]
                if link["id"] == key
            ],
        }

    def validate(self):
        for key, node in self.groups["results"].items():
            if node["status"] == "verified":
                self.values(key)
        for key in self.groups["claims"]:
            self.claim(key)
        for key, node in self.groups["materials"].items():
            require(self.path(node["path"]).is_file(), f"{key}: missing {node['path']}")


def run(args):
    try:
        graph = Graph(
            content(REPO_ROOT / "research/graph.json"),
            {"entries": []},
        )
        if args.action == "check":
            require(args.key is None, "check accepts no ID")
            graph.validate()
            print("Verified pinned evidence and material paths.")
        else:
            require(args.key, "specify a claim ID, node ID or evidence path")
            key = args.key
            if args.action == "impact" and key not in graph.nodes and key not in graph.files:
                key = str(PROJECT_PREFIX / key)
            if args.action == "claim":
                value = graph.claim(key)
            else:
                affected = graph.impact(key)
                materials = graph.groups["materials"]
                value = {
                    "affected": affected,
                    "materials": {item: materials[item] for item in affected if item in materials},
                }
            print(json.dumps(value, indent=2, ensure_ascii=False))
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        print(f"evidence check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "claim", "impact"))
    parser.add_argument("key", nargs="?")
    sys.exit(run(parser.parse_args()))
