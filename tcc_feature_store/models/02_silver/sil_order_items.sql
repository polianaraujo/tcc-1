-- Aqui vamos garantir que preços e fretes não sejam nulos ou negativos, além de formatar a data de limite de envio.


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