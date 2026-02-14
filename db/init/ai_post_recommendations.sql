create extension if not exists vector;

create table if not exists ai_post_recommendations (
  "postId" bigint primary key,
  "situationEmbedding" vector(768) not null,
  "decisionEmbedding" vector(768) not null,
  "tagId" bigint[] not null default '{}',
  "tagName" text[] not null default '{}',
  "updatedAt" timestamptz not null default now()
);

-- Cosine-similarity ANN indexes (tune lists by dataset size)
create index if not exists idx_ai_post_recommendations_situation_ivfflat
  on ai_post_recommendations using ivfflat ("situationEmbedding" vector_cosine_ops)
  with (lists = 100);

create index if not exists idx_ai_post_recommendations_decision_ivfflat
  on ai_post_recommendations using ivfflat ("decisionEmbedding" vector_cosine_ops)
  with (lists = 100);
