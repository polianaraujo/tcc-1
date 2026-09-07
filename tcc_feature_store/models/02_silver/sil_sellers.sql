-- Esta tabela é vital, pois o vendedor será o foco do nosso Data Leakage. Vamos apenas padronizar os nomes.


{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('bronze', 'RAW_olist_sellers_dataset') }}
),

renamed AS (
    SELECT
        seller_id,
        seller_zip_code_prefix AS seller_zip_code,
        UPPER(seller_city) AS seller_city,
        UPPER(seller_state) AS seller_state
    FROM source
)

SELECT * FROM renamed