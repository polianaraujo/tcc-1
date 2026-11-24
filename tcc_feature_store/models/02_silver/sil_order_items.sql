-- Renomeia as colunas, converte preço e frete para FLOAT e mantém apenas itens com preço positivo
-- (o frete não é filtrado: o valor 0 é válido).
-- Não há deduplicação: um pedido pode ter vários itens do mesmo vendedor (repetição legítima).
-- A Gold colapsa isso com SELECT DISTINCT (order_id, seller_id) em gold_pedidos_vendedor.


{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('bronze', 'RAW_olist_order_items_dataset') }}
),

renamed AS (
    SELECT
        order_id,
        order_item_id,
        product_id,
        seller_id,
        -- Removendo o TRY_CAST
        shipping_limit_date AS shipping_limit_at,
        CAST(price AS FLOAT) AS item_price,
        CAST(freight_value AS FLOAT) AS freight_price
    FROM source
)

SELECT *
FROM renamed
WHERE item_price > 0