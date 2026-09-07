{{ config(
    materialized='table',
    schema='gold'
) }}

WITH seller_orders AS (
    -- 1. Juntamos todas as tabelas da Silver para ter o cenário completo de cada pedido
    SELECT 
        o.order_id,
        i.seller_id,
        o.order_purchase_at,
        -- Regra de negócio: Se entregou depois da data estimada, marca 1 (teve atraso)
        CASE WHEN o.order_delivered_at > o.order_estimated_delivery_at THEN 1 ELSE 0 END AS is_delayed,
        r.review_score
    FROM {{ ref('sil_orders') }} o
    JOIN {{ ref('sil_order_items') }} i ON o.order_id = i.order_id
    LEFT JOIN {{ ref('sil_reviews') }} r ON o.order_id = r.order_id
    WHERE o.order_purchase_at IS NOT NULL
),

running_features AS (
    -- 2. Aqui usamos Window Functions para calcular o histórico ACUMULADO até o momento de cada pedido
    SELECT 
        order_id,      -- Corrigido: Agora o pedido vai para a tabela final
        is_delayed,    -- Corrigido: Agora o nosso target vai para a tabela final
        seller_id,
        order_purchase_at AS feature_timestamp, -- Esta é a data que o ASOF JOIN vai olhar!
        
        -- Feature 1: Conta quantos pedidos o vendedor já teve até esta data
        COUNT(order_id) OVER (
            PARTITION BY seller_id 
            ORDER BY order_purchase_at 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS seller_order_volume,
        
        -- Feature 2: Soma quantos atrasos ele acumulou até esta data
        SUM(is_delayed) OVER (
            PARTITION BY seller_id 
            ORDER BY order_purchase_at 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS seller_delay_count,
        
        -- Feature 3: Tira a média das notas recebidas até esta data
        AVG(review_score) OVER (
            PARTITION BY seller_id 
            ORDER BY order_purchase_at 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS seller_avg_review_score

    FROM seller_orders
)

-- 3. Entregamos a Feature Store pronta
SELECT * FROM running_features