{% snapshot snap_seller_evolution %}

{{ config(
    target_database='tcc_lakehouse',
    target_schema='silver',
    unique_key='seller_id',
    strategy='check',
    check_cols=['review_score', 'order_status']
) }}

WITH seller_reviews AS (
    SELECT 
        s.seller_id,
        s.seller_city,
        o.order_purchase_at,
        o.order_status,
        r.review_score,
        r.review_created_at
    FROM {{ ref('sil_sellers') }} s
    JOIN {{ ref('sil_order_items') }} i ON s.seller_id = i.seller_id
    JOIN {{ ref('sil_orders') }} o ON i.order_id = o.order_id
    JOIN {{ ref('sil_reviews') }} r ON o.order_id = r.order_id
)

SELECT * FROM seller_reviews

{% endsnapshot %}