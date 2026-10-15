-- gold_fs_seller_review.sql
-- Linha do tempo da NOTA MÉDIA de avaliações do vendedor.
-- Disponibilidade: a nota de um pedido só é conhecida quando o cliente RESPONDE
-- a avaliação (review_answer_timestamp).

{{ config(materialized='table', schema='gold') }}

WITH eventos AS (
    SELECT
        seller_id,
        review_answered_at AS feature_timestamp,
        SUM(review_score)  AS soma_notas,
        COUNT(*)           AS n_avaliacoes
    FROM {{ ref('gold_pedidos_vendedor') }}
    WHERE review_score IS NOT NULL
      AND review_answered_at IS NOT NULL
    GROUP BY seller_id, review_answered_at
)

SELECT
    seller_id,
    feature_timestamp,
    SUM(soma_notas) OVER (
        PARTITION BY seller_id ORDER BY feature_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
    / SUM(n_avaliacoes) OVER (
        PARTITION BY seller_id ORDER BY feature_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS seller_avg_review_score
FROM eventos
