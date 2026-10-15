-- gold_fs_seller_delay.sql
-- Linha do tempo da CONTAGEM DE ATRASOS do vendedor.
-- Disponibilidade: o atraso de um pedido só é conhecido quando ele é ENTREGUE.
-- (Escolha conservadora: usar um instante posterior ao real nunca gera vazamento.)

{{ config(materialized='table', schema='gold') }}

WITH eventos AS (
    SELECT
        seller_id,
        order_delivered_at AS feature_timestamp,
        SUM(is_delayed)    AS n_atrasos
    FROM {{ ref('gold_pedidos_vendedor') }}
    WHERE order_delivered_at IS NOT NULL
    GROUP BY seller_id, order_delivered_at
)

SELECT
    seller_id,
    feature_timestamp,
    SUM(n_atrasos) OVER (
        PARTITION BY seller_id ORDER BY feature_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS seller_delay_count
FROM eventos
