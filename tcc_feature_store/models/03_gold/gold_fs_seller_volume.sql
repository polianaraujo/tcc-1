-- gold_fs_seller_volume.sql
-- Linha do tempo do VOLUME de pedidos do vendedor.
-- Disponibilidade: um pedido passa a contar no volume no instante da COMPRA.
-- Conta todos os pedidos recebidos (inclusive os que depois foram cancelados),
-- pois no instante da compra ainda não se sabe o desfecho de nenhum deles.

{{ config(materialized='table', schema='gold') }}

WITH eventos AS (
    -- Agrupa pedidos do mesmo vendedor no mesmo instante: garante uma única
    -- versão por (seller_id, feature_timestamp) e torna as junções determinísticas
    SELECT
        seller_id,
        order_purchase_at AS feature_timestamp,
        COUNT(*)          AS n_pedidos
    FROM {{ ref('gold_pedidos_vendedor') }}
    GROUP BY seller_id, order_purchase_at
)

SELECT
    seller_id,
    feature_timestamp,
    SUM(n_pedidos) OVER (
        PARTITION BY seller_id ORDER BY feature_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS seller_order_volume
FROM eventos
