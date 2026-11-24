-- Este modelo limpa a tabela de pedidos: renomeia as colunas de data e remove linhas sem data de compra
-- (um pedido sem data de compra quebra qualquer Feature Store temporal).
-- Os nulos de entrega são mantidos de propósito: a Gold usa order_delivered_at IS NULL para excluir pedidos sem desfecho observável.
-- Não há conversão de tipo (a Bronze já traz as datas como TIMESTAMP_NTZ) nem deduplicação (order_id é único).


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