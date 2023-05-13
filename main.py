import utils

def main():
    day_type_user='1d'
    mark_type_user='Close'
    
    df=utils.calc_log_return(utils.load_data_yahoo(tickerSymbol='META', period=day_type_user),mark_type=mark_type_user)
    n=utils.pred_interval(data=df,day_type=day_type_user)
    utils.plot_logr_price(data=df,mark_type=mark_type_user)
    print(utils.descriptive_stats(df))

    #Calling models
    utils.mc_model(data=df,sim=10000, n=n, day_type=day_type_user, mark_type=mark_type_user)


if __name__ == "__main__":
    main()
