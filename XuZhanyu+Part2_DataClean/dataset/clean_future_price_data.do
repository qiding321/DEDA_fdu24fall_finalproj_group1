********************************************************************************
********       Predicting Future Price by Using Machine Learning   *************

*** Import Data File:
	global dir "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project/"
	import delimited "${dir}ec_fut.csv", encoding(UTF-8) clear

*** Create Date Data:
	gen trade_date_temp = date(trade_date, "YMD")
	format trade_date_temp %td
	drop trade_date
	rename trade_date_temp trade_date
	label var trade_date "Trading Date"
	gen trade_week = wofd(trade_date)
	format trade_week %tw
	rename trade_week week
	
	gen last_trade_date_temp = date(last_trade_date, "YMD")
	format last_trade_date_temp %td
	drop last_trade_date
	rename last_trade_date_temp last_trade_date
	label var last_trade_date "Last Trading Date"

	gen update_time_temp = clock(update_time, "YMDhms")
	format update_time_temp %tc
	drop update_time
	rename update_time_temp update_time
	label var update_time "Update Time"
	
*** Converting String Variables into Numeric Variables:
	/// We don't need "集运指数（欧线"）in variable of "sec_short_name":
	gen end_date = substr(sec_short_name , -4, .)
	gen y = 20
	egen end_date_temp= concat(y end_date)
	drop y
	gen end_date_temp_temp = mdy(real(substr(end_date_temp, 5, 2)), 1, real(substr(end_date_temp, 1, 4)))
	format end_date_temp_temp %td
	drop end_date_temp end_date
	rename end_date_temp_temp end_date	
	
	foreach x of varlist ticker_symbol exchange_cd sec_short_name contract_object contract_mark {
	encode `x', gen(`x'_temp)
	drop `x'
	rename `x'_temp `x'
	}
	replace ticker_symbol = 2404 if ticker_symbol == 1
	replace ticker_symbol = 2406 if ticker_symbol == 2
	replace ticker_symbol = 2408 if ticker_symbol == 3
	replace ticker_symbol = 2410 if ticker_symbol == 4
	replace ticker_symbol = 2412 if ticker_symbol == 5
	replace ticker_symbol = 2502 if ticker_symbol == 6
	replace ticker_symbol = 2504 if ticker_symbol == 7
	replace ticker_symbol = 2506 if ticker_symbol == 8
	replace ticker_symbol = 2508 if ticker_symbol == 9

	label var ticker_symbol "Ticker Symbol"
	label var sec_short_name "Abbreviation of contract"
	keep if maincon == 1
	order trade_date
	label var pre_settl_price "Pre Settle Price"
	label var pre_close_price "Pre Close Price"
	label var open_price "Open Price"
	label var highest_price "Highest Price"
	label var lowest_price "Lowest Price"
	label var settl_price "Settle Price"
	label var close_price "Close Price"
	label var turnover_vol "Turnover Volume"
	label var turnover_value "Turnover Value"
	label var open_int "Open Inventory"
	

*** Save Dataset:
	export excel using "${dir}future.xlsx", firstrow(variables) replace
	cd "${dir}"
	save future, replace
	
	export delimited using "${dir}future.csv", replace

	
	
	
	
	
	
	
	
	
