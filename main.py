import modules_functions

def main():
    day_type_user='1d'
    mark_type_user='Close'
    
    df=modules_functions.calc_log_return(modules_functions.load_data_yahoo(tickerSymbol='META', period=day_type_user),mark_type=mark_type_user)
    n=modules_functions.pred_interval(data=df,day_type=day_type_user)
    modules_functions.plot_logr_price(data=df,mark_type=mark_type_user)
    print(modules_functions.descriptive_stats(df))

    #Calling models
    modules_functions.mc_model(data=df,sim=10000, n=n, day_type=day_type_user, mark_type=mark_type_user)


if __name__ == "__main__":
    main()
