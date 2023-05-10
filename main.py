import modules_functions

def main():
    day_type_user='1d'
    df=modules_functions.calc_log_return(modules_functions.load_data_yahoo(tickerSymbol='META'), period=day_type_user)
    n=modules_functions.pred_interval(data=df,day_type=day_type_user)
    modules_functions.plot_logr_price(data=df)
    print(modules_functions.descriptive_stats(df))

    n=pred_interval(da)
    #Calling models
    modules_functions.mc_model(data=df, n=n, day_type=day_type_user)


if __name__ == "__main__":
    main()
