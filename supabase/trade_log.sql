-- Trade history for the dashboard (trade_history.py). Already applied to the
-- "Fantasy Football Dashboard" Supabase project; kept here to recreate it elsewhere.
create table public.trade_log (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  league_id bigint not null,
  season int not null,
  week int,
  source text not null check (source in ('app','espn')),
  espn_key text unique,
  from_team text not null,
  to_team text not null,
  sends jsonb not null default '[]'::jsonb,
  receives jsonb not null default '[]'::jsonb,
  status text not null check (status in ('sent','accepted','rejected','countered','completed')),
  predicted_accept double precision,
  predicted_mine double precision,
  tier text,
  notes text
);
create index trade_log_league_idx on public.trade_log (league_id, season);
-- The app connects server-side with the secret key, which bypasses RLS. No public
-- policies on purpose: the publishable key can neither read nor write the log.
alter table public.trade_log enable row level security;
