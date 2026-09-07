-- Este modelo limpa a tabela de pedidos. Vamos remover linhas onde as datas cruciais estão nulas (porque um pedido sem data quebra qualquer Feature Store temporal) e garantir que tudo seja TIMESTAMP.


{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('bronze', 'RAW_olist_orders_dataset') }}
),

renamed AS (
    SELECT
        order_id,
        customer_id,
        order_status,
        -- Removendo os TRY_CAST pois já são timestamps
        order_purchase_timestamp AS order_purchase_at,
        order_approved_at,
        order_delivered_carrier_date AS order_carrier_at,
        order_delivered_customer_date AS order_delivered_at,
        order_estimated_delivery_date AS order_estimated_delivery_at
    FROM source
)

SELECT *
FROM renamed
WHERE order_purchase_at IS NOT NULL