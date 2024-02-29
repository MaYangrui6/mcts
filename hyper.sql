select 
	s_store_name,
	i_item_desc,
	sc.revenue,
	i_current_price,
	i_wholesale_cost,
	i_brand
 from store, item,
     (select ss_store_sk, avg(revenue) as ave
 	from
 	    (select  ss_store_sk, ss_item_sk, 
 		     sum(ss_sales_price) as revenue
 		from store_sales, date_dim
 		where ss_sold_date_sk = d_date_sk and d_month_seq between 1180 and 1180+11
 		group by ss_store_sk, ss_item_sk) sa
 	group by ss_store_sk) sb,
     (select  ss_store_sk, ss_item_sk, sum(ss_sales_price) as revenue
 	from store_sales, date_dim
 	where ss_sold_date_sk = d_date_sk and d_month_seq between 1180 and 1180+11
 	group by ss_store_sk, ss_item_sk) sc
 where sb.ss_store_sk = sc.ss_store_sk and 
       sc.revenue <= 0.1 * sb.ave and
       s_store_sk = sc.ss_store_sk and
       i_item_sk = sc.ss_item_sk
 order by s_store_name, i_item_desc
limit 100;

select avg(ss_quantity)
       ,avg(ss_ext_sales_price)
       ,avg(ss_ext_wholesale_cost)
       ,sum(ss_ext_wholesale_cost)
 from store_sales
     ,store
     ,customer_demographics
     ,household_demographics
     ,customer_address
     ,date_dim
 where s_store_sk = ss_store_sk
 and  ss_sold_date_sk = d_date_sk and d_year = 2001
 and((ss_hdemo_sk=hd_demo_sk
  and cd_demo_sk = ss_cdemo_sk
  and cd_marital_status = 'U'
  and cd_education_status = 'Secondary'
  and ss_sales_price between 100.00 and 150.00
  and hd_dep_count = 3   
     )or
     (ss_hdemo_sk=hd_demo_sk
  and cd_demo_sk = ss_cdemo_sk
  and cd_marital_status = 'M'
  and cd_education_status = 'College'
  and ss_sales_price between 50.00 and 100.00   
  and hd_dep_count = 1
     ) or 
     (ss_hdemo_sk=hd_demo_sk
  and cd_demo_sk = ss_cdemo_sk
  and cd_marital_status = 'S'
  and cd_education_status = 'Unknown'
  and ss_sales_price between 150.00 and 200.00 
  and hd_dep_count = 1  
     ))
 and((ss_addr_sk = ca_address_sk
  and ca_country = 'United States'
  and ca_state in ('LA', 'VA', 'KY')
  and ss_net_profit between 100 and 200  
     ) or
     (ss_addr_sk = ca_address_sk
  and ca_country = 'United States'
  and ca_state in ('AR', 'MD', 'IN')
  and ss_net_profit between 150 and 300  
     ) or
     (ss_addr_sk = ca_address_sk
  and ca_country = 'United States'
  and ca_state in ('FL', 'MS', 'TX')
  and ss_net_profit between 50 and 250  
     ))
;

-- select  count(*) from (
--     select distinct c_last_name, c_first_name, d_date
--     from store_sales, date_dim, customer
--           where store_sales.ss_sold_date_sk = date_dim.d_date_sk
--       and store_sales.ss_customer_sk = customer.c_customer_sk
--       and d_month_seq between 1210 and 1210 + 11
--   intersect
--     select distinct c_last_name, c_first_name, d_date
--     from catalog_sales, date_dim, customer
--           where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
--       and catalog_sales.cs_bill_customer_sk = customer.c_customer_sk
--       and d_month_seq between 1210 and 1210 + 11
--   intersect
--     select distinct c_last_name, c_first_name, d_date
--     from web_sales, date_dim, customer
--           where web_sales.ws_sold_date_sk = date_dim.d_date_sk
--       and web_sales.ws_bill_customer_sk = customer.c_customer_sk
--       and d_month_seq between 1210 and 1210 + 11
-- ) hot_cust
-- limit 100;

-- select  i_item_id
--        ,i_item_desc 
--        ,i_category 
--        ,i_class 
--        ,i_current_price
--        ,sum(cs_ext_sales_price) as itemrevenue 
--        ,sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over
--            (partition by i_class) as revenueratio
--  from	catalog_sales
--      ,item 
--      ,date_dim
--  where cs_item_sk = i_item_sk 
--    and i_category in ('Children', 'Jewelry', 'Home')
--    and cs_sold_date_sk = d_date_sk
--  and d_date between cast('1998-04-22' as date) 
--  				and (cast('1998-04-22' as date) + interval '30' day)
--  group by i_item_id
--          ,i_item_desc 
--          ,i_category
--          ,i_class
--          ,i_current_price
--  order by i_category
--          ,i_class
--          ,i_item_id
--          ,i_item_desc
--          ,revenueratio
-- limit 100;

-- select 
--   c_last_name,c_first_name,substr(s_city,1,30),ss_ticket_number,amt,profit
--   from
--    (select ss_ticket_number
--           ,ss_customer_sk
--           ,store.s_city
--           ,sum(ss_coupon_amt) amt
--           ,sum(ss_net_profit) profit
--     from store_sales,date_dim,store,household_demographics
--     where store_sales.ss_sold_date_sk = date_dim.d_date_sk
--     and store_sales.ss_store_sk = store.s_store_sk  
--     and store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
--     and (household_demographics.hd_dep_count = 8 or household_demographics.hd_vehicle_count > 0)
--     and date_dim.d_dow = 1
--     and date_dim.d_year in (1998,1998+1,1998+2) 
--     and store.s_number_employees between 200 and 295
--     group by ss_ticket_number,ss_customer_sk,ss_addr_sk,store.s_city) ms,customer
--     where ss_customer_sk = c_customer_sk
--  order by c_last_name,c_first_name,substr(s_city,1,30), profit
-- limit 100;

-- select  i_item_id, 
--         avg(ss_quantity) agg1,
--         avg(ss_list_price) agg2,
--         avg(ss_coupon_amt) agg3,
--         avg(ss_sales_price) agg4 
--  from store_sales, customer_demographics, date_dim, item, promotion
--  where ss_sold_date_sk = d_date_sk and
--        ss_item_sk = i_item_sk and
--        ss_cdemo_sk = cd_demo_sk and
--        ss_promo_sk = p_promo_sk and
--        cd_gender = 'F' and 
--        cd_marital_status = 'M' and
--        cd_education_status = '4 yr Degree' and
--        (p_channel_email = 'N' or p_channel_event = 'N') and
--        d_year = 2002 
--  group by i_item_id
--  order by i_item_id
--  limit 100;
 
--  select  ca_zip, ca_state, sum(ws_sales_price)
--  from web_sales, customer, customer_address, date_dim, item
--  where ws_bill_customer_sk = c_customer_sk
--  	and c_current_addr_sk = ca_address_sk 
--  	and ws_item_sk = i_item_sk 
--  	and ( substr(ca_zip,1,5) in ('85669', '86197','88274','83405','86475', '85392', '85460', '80348', '81792')
--  	      or 
--  	      i_item_id in (select i_item_id
--                              from item
--                              where i_item_sk in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
--                              )
--  	    )
--  	and ws_sold_date_sk = d_date_sk
--  	and d_qoy = 1 and d_year = 2000
--  group by ca_zip, ca_state
--  order by ca_zip, ca_state
--  limit 100;