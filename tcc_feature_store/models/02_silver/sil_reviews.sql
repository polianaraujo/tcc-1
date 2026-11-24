-- As avaliações serão usadas depois para calcular, por exemplo, a "nota média do vendedor no último mês".
-- Não há deduplicação: um pedido pode ter mais de uma avaliação (repetição legítima, com notas possivelmente diferentes).
-- A Gold agrega por order_id (AVG da nota e MAX do instante da resposta) em gold_pedidos_vendedor.


{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('bronze', 'RAW_olist_order_reviews_dataset') }}
),

renamed AS (
    SELECT
        review_id,
        order_id,
        CAST(review_score AS INTEGER) AS review_score,
        review_comment_title,
        review_comment_message,
        -- Removendo os TRY_CAST
        review_creation_date AS review_created_at,
        review_answer_timestamp AS review_answered_at
    FROM source
)

SELECT *
FROM renamed
WHERE review_score BETWEEN 1 AND 5