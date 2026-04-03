from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProductRecord:
    id: str
    category: str
    subcategory: str
    title: str
    brand: str
    price: float
    colors: list[str]
    sizes: list[str]
    parameters: dict[str, Any]
    rating: float
    shop_name: str
    after_sales: dict[str, Any]
    description: str
    image: str

    @property
    def image_path(self) -> str:
        return self.image

    @property
    def category_slug(self) -> str:
        return Path(self.image).parts[-3]

    @property
    def subcategory_slug(self) -> str:
        return Path(self.image).parts[-2]


@dataclass(slots=True)
class SubcategoryQuery:
    key: str
    category_slug: str
    category: str
    canonical_subcategory: str
    slug_text: str
    query_text: str
    image_paths: list[str]
    positive_indices: list[int]

    @property
    def slug(self) -> str:
        return self.key

    @property
    def canonical_zh(self) -> str:
        return self.canonical_subcategory

    @property
    def relevant_indices(self) -> list[int]:
        return self.positive_indices


def load_products(dataset_dir: Path) -> list[ProductRecord]:
    products_path = dataset_dir / "products.jsonl"
    records: list[ProductRecord] = []
    with products_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            records.append(ProductRecord(**payload))
    return records


def slug_to_text(slug: str) -> str:
    return slug.replace("_", " ").strip()


def stringify_value(value: Any) -> str:
    if isinstance(value, list):
        return "、".join(map(str, value))
    if isinstance(value, dict):
        return "；".join(f"{key}: {val}" for key, val in value.items())
    return str(value)


def build_query_text(
    product: ProductRecord,
    mode: str = "rich_zh",
) -> str:
    if mode == "description_only":
        return product.description.strip()

    if mode == "title_brand_desc":
        return "；".join(
            part
            for part in [
                f"商品标题: {product.title}",
                f"品牌: {product.brand}",
                f"商品描述: {product.description}",
            ]
            if part
        )

    if mode != "rich_zh":
        raise ValueError(f"Unsupported query mode: {mode}")

    fields = [
        ("商品ID", product.id),
        ("一级类目", product.category),
        ("二级类目", product.subcategory),
        ("标题", product.title),
        ("品牌", product.brand),
        ("价格", f"{product.price:.2f}"),
        ("颜色", product.colors),
        ("尺码/规格", product.sizes),
        ("商品参数", product.parameters),
        ("店铺", product.shop_name),
        ("评分", product.rating),
        ("售后", product.after_sales),
        ("商品描述", product.description),
    ]
    return "\n".join(
        f"{name}: {stringify_value(value)}"
        for name, value in fields
        if value not in (None, "", [], {})
    )


def build_subcategory_queries(
    products: list[ProductRecord],
    mode: str = "canonical_zh",
) -> list[SubcategoryQuery]:
    grouped: dict[str, list[tuple[int, ProductRecord]]] = defaultdict(list)
    for index, product in enumerate(products):
        image_path = Path(product.image_path)
        folder_key = image_path.parent.name
        grouped[folder_key].append((index, product))

    queries: list[SubcategoryQuery] = []
    for key in sorted(grouped):
        members = grouped[key]
        category_slug = Path(members[0][1].image_path).parent.parent.name
        category = Counter(product.category for _, product in members).most_common(1)[0][0]
        canonical_subcategory = Counter(
            product.subcategory for _, product in members
        ).most_common(1)[0][0]
        slug_text = slug_to_text(key)

        if mode == "canonical_zh":
            query_text = canonical_subcategory
        elif mode == "canonical_zh_with_category":
            query_text = f"{category} {canonical_subcategory}"
        elif mode == "slug_en":
            query_text = slug_text
        elif mode == "slug_en_with_category":
            query_text = f"{slug_to_text(category_slug)} {slug_text}"
        else:
            raise ValueError(f"Unsupported subcategory query mode: {mode}")

        queries.append(
            SubcategoryQuery(
                key=key,
                category_slug=category_slug,
                category=category,
                canonical_subcategory=canonical_subcategory,
                slug_text=slug_text,
                query_text=query_text,
                image_paths=[product.image_path for _, product in members],
                positive_indices=[index for index, _ in members],
            )
        )

    return queries
