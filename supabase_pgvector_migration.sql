-- Supabase pgvector migration for document chunk retrieval.
-- Default dimension here is 384, which matches sentence-transformers/all-MiniLM-L6-v2.
-- If you switch to a different embedding model such as Gemini embeddings,
-- update the vector dimension before running this migration and re-index all chunks.

create extension if not exists vector with schema extensions;

alter table public.document_chunks
add column if not exists embedding extensions.vector(384);

create index if not exists document_chunks_embedding_hnsw_idx
on public.document_chunks
using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.match_document_chunks(
  query_embedding extensions.vector(384),
  match_count int,
  filter_user_id uuid,
  filter_provider text
)
returns table (
  id uuid,
  document_id uuid,
  chunk_index bigint,
  kind text,
  page_number bigint,
  char_count bigint,
  content text,
  summary text,
  vector_id text,
  created_at timestamptz,
  source text,
  distance float
)
language sql
as $$
  select
    dc.id,
    dc.document_id,
    dc.chunk_index,
    dc.kind,
    dc.page_number,
    dc.char_count,
    dc.content,
    dc.summary,
    dc.vector_id,
    dc.created_at,
    d.filename as source,
    dc.embedding <=> query_embedding as distance
  from public.document_chunks dc
  join public.documents d on d.id = dc.document_id
  where d.user_id = filter_user_id
    and d.provider = filter_provider
    and dc.embedding is not null
  order by dc.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;
