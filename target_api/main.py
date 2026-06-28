"""
Target FastAPI app — a simple Blog Posts + Comments API.
Intentional bugs are injected (marked with # BUG) so Week 2 triage has real failures to classify.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime

app = FastAPI(
    title="Blog API",
    description="A simple Blog Posts and Comments API for API test generation demo.",
    version="1.0.0",
)

# ── In-memory stores ──────────────────────────────────────────────────────────
posts: dict[str, dict] = {}
comments: dict[str, dict] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────
class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)


class PostResponse(BaseModel):
    id: str
    title: str
    body: str
    author: str
    created_at: str


class CommentCreate(BaseModel):
    post_id: str
    body: str = Field(..., min_length=1)
    commenter: str = Field(..., min_length=1)


class CommentResponse(BaseModel):
    id: str
    post_id: str
    body: str
    commenter: str
    created_at: str


# ── Posts endpoints ───────────────────────────────────────────────────────────
@app.post("/posts", response_model=PostResponse, status_code=201)
def create_post(payload: PostCreate):
    post_id = str(uuid.uuid4())
    post = {
        "id": post_id,
        "title": payload.title,
        "body": payload.body,
        "author": payload.author,
        "created_at": datetime.utcnow().isoformat(),
    }
    posts[post_id] = post
    return post


@app.get("/posts", response_model=list[PostResponse])
def list_posts(limit: int = Query(default=10, ge=1, le=100)):
    # BUG-001: limit param is silently ignored — always returns all posts
    return list(posts.values())


@app.get("/posts/{post_id}", response_model=PostResponse)
def get_post(post_id: str):
    if post_id not in posts:
        raise HTTPException(status_code=404, detail="Post not found")
    return posts[post_id]


@app.put("/posts/{post_id}", response_model=PostResponse)
def update_post(post_id: str, payload: PostCreate):
    if post_id not in posts:
        raise HTTPException(status_code=404, detail="Post not found")
    posts[post_id].update(
        title=payload.title,
        body=payload.body,
        author=payload.author,
    )
    return posts[post_id]


@app.delete("/posts/{post_id}")
def delete_post(post_id: str):
    if post_id not in posts:
        raise HTTPException(status_code=404, detail="Post not found")
    del posts[post_id]
    # BUG-002: returns 200 instead of the conventional 204
    return {"deleted": True}


# ── Comments endpoints ────────────────────────────────────────────────────────
@app.post("/comments", response_model=CommentResponse, status_code=201)
def create_comment(payload: CommentCreate):
    # BUG-003: does NOT validate that payload.post_id actually exists
    comment_id = str(uuid.uuid4())
    comment = {
        "id": comment_id,
        "post_id": payload.post_id,
        "body": payload.body,
        "commenter": payload.commenter,
        "created_at": datetime.utcnow().isoformat(),
    }
    comments[comment_id] = comment
    return comment


@app.get("/posts/{post_id}/comments", response_model=list[CommentResponse])
def list_comments_for_post(post_id: str):
    if post_id not in posts:
        raise HTTPException(status_code=404, detail="Post not found")
    return [c for c in comments.values() if c["post_id"] == post_id]


@app.get("/comments/{comment_id}", response_model=CommentResponse)
def get_comment(comment_id: str):
    if comment_id not in comments:
        raise HTTPException(status_code=404, detail="Comment not found")
    return comments[comment_id]


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "posts": len(posts), "comments": len(comments)}