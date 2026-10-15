-- gold_pedidos_vendedor.sql
-- Base de eventos da Feature Store: um registro por (pedido, vendedor), com os
-- instantes em que cada informação passa a existir no mundo real.

{{ config(materialized='table', schema='gold') }}

WITH pedido_vendedor AS (
    -- DISTINCT: um pedido com vários itens do MESMO vendedor conta uma única vez
    SELECT DISTINCT
        o.order_id,
        i.seller_id,
        o.order_status,
        o.order_purchase_at,
        o.order_delivered_at,
        o.order_estimated_delivery_at
    FROM {{ ref('sil_orders') }} o
    JOIN {{ ref('sil_order_items') }} i ON o.order_id = i.order_id
    WHERE o.order_purchase_at IS NOT NULL
),

review_por_pedido AS (
    -- Olist ocasionalmente tem mais de uma avaliação por pedido:
    -- nota média e o instante da ÚLTIMA resposta (quando a nota final passa a ser conhecida)
    SELECT
        order_id,
        AVG(review_score)        AS review_score,
        MAX(review_answered_at)  AS review_answered_at
    FROM {{ ref('sil_reviews') }}
    GROUP BY order_id
)

SELECT
    pv.order_id,
    pv.seller_id,
    pv.order_status,
    pv.order_purchase_at,
    pv.order_delivered_at,
    pv.order_estimated_delivery_at,
    -- Atraso = entregue em DIA posterior ao prometido. A data estimada do Olist é
    -- sempre 00:00:00; comparar timestamps marcaria como atraso a entrega feita no
    -- próprio dia prometido.
    CASE
        WHEN pv.order_delivered_at IS NULL THEN NULL
        WHEN pv.order_delivered_at::DATE > pv.order_estimated_delivery_at::DATE THEN 1
        ELSE 0
    END AS is_delayed,
    rp.review_score,
    rp.review_answered_at
FROM pedido_vendedor pv
LEFT JOIN review_por_pedido rp ON pv.order_id = rp.order_id
